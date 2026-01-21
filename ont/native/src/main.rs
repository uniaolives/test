use clap::Parser;
use ontology_lang::cli::{Cli, Commands};
use ontology_lang::audit::evm_audit::{EVMAuditor, install_panic_hook, AuditUpdate};
use ontology_lang::{InvariantWitness, DeploymentTarget, ProductionAuditor};
use ethers::providers::{Provider, Http};
use ethers::prelude::*;
use std::sync::Arc;
use log::{info, error};
use tokio::time::Duration;
use tokio::sync::broadcast;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Instalar hook de pânico para auditoria
    install_panic_hook();

    // Configurar logging
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Compile { input, output, target } => {
            ontology_lang::compile(&input, output.as_deref(), &target)?;
        }

        Commands::Deploy { bytecode, rpc_url, private_key } => {
            ontology_lang::deploy(&bytecode, &rpc_url, &private_key).await?;
        }

        Commands::Audit { contract, quantum_seed, rpc, private_key, interval, mobile, daemon } => {
            if daemon {
                // Executar em background usando sh -c para tratar redirecionamentos
                let mut cmd_args = vec![
                    "audit".to_string(),
                    "--contract".to_string(), contract,
                    "--quantum-seed".to_string(), quantum_seed,
                    "--rpc".to_string(), rpc,
                    "--interval".to_string(), interval.to_string(),
                ];
                if let Some(pk) = private_key {
                    cmd_args.push("--private-key".to_string());
                    cmd_args.push(pk);
                }
                if mobile {
                    cmd_args.push("--mobile".to_string());
                }

                let full_cmd = format!("nohup onto {} > logs/audit_daemon.log 2>&1 &", cmd_args.join(" "));

                std::process::Command::new("sh")
                    .arg("-c")
                    .arg(full_cmd)
                    .status()
                    .map_err(|e| format!("Failed to start daemon: {}", e))?;

                println!("✅ Audit daemon started: logs/audit_daemon.log");
            } else {
                // Inicializar auditoria
                info!("Starting continuous audit for contract: {}", contract);

                // Converter seed hex para bytes
                let seed_bytes = hex::decode(quantum_seed.trim_start_matches("0x"))
                    .expect("Invalid hex quantum seed");

                if seed_bytes.len() != 32 {
                    panic!("Quantum seed must be 32 bytes");
                }

                let mut seed_array = [0u8; 32];
                seed_array.copy_from_slice(&seed_bytes);

                // Criar canal de broadcast para o dashboard
                let (tx, _rx) = broadcast::channel::<AuditUpdate>(100);

                // Iniciar servidor WebSocket se dashboard estiver ativo
                if std::path::Path::new("./dashboard").exists() {
                    let tx_clone = tx.clone();
                    tokio::spawn(async move {
                        start_ws_server(tx_clone).await;
                    });
                }

                // Criar cliente Ethereum
                let provider = Provider::<Http>::try_from(rpc.clone())
                    .expect("Failed to create Ethereum provider");

                let address: Address = contract.parse()
                    .expect("Invalid contract address");

                if let Some(pk) = private_key {
                    let wallet: LocalWallet = pk.parse().expect("Invalid private key");
                    let client = SignerMiddleware::new(provider, wallet.with_chain_id(31337u64));
                    let client_arc = Arc::new(client);

                    let auditor = EVMAuditor::new(
                        client_arc,
                        address,
                        seed_array,
                        tx,
                    );

                    println!("✅ Audit loop started with SIGNER. Interval: {} seconds", interval);
                    auditor.start().await;
                } else {
                    let client_arc = Arc::new(provider);
                    let auditor = EVMAuditor::new(
                        client_arc,
                        address,
                        seed_array,
                        tx,
                    );

                    println!("✅ Audit loop started (READ-ONLY). Interval: {} seconds", interval);
                    auditor.start().await;
                }
            }
        }

        Commands::AuditStatus { contract, rpc_url } => {
            println!("Audit status for: {}", contract);
            println!("RPC: {}", rpc_url);
            println!("");
            println!("Feature coming soon...");
        }

        Commands::GemSimulator { geometry, matter, duration_steps, hubble_parameter, output_file } => {
            ontology_lang::gem_simulator::run_simulator(
                geometry,
                matter,
                duration_steps,
                hubble_parameter,
                output_file
            ).await?;
        }
    }

    Ok(())
}

async fn start_ws_server(tx: broadcast::Sender<AuditUpdate>) {
    use warp::Filter;
    use futures_util::{StreamExt, SinkExt};

    let audit_route = warp::path("audit")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let tx = tx.clone();
            ws.on_upgrade(move |socket| {
                let mut rx = tx.subscribe();
                async move {
                    let (mut ws_tx, _) = socket.split();
                    while let Ok(update) = rx.recv().await {
                        if let Ok(json) = serde_json::to_string(&update) {
                            if let Err(_) = ws_tx.send(warp::ws::Message::text(json)).await {
                                break;
                            }
                        }
                    }
                }
            })
        });

    println!("📡 WebSocket server starting on port 8081...");
    warp::serve(audit_route).run(([0, 0, 0, 0], 8081)).await;
}
