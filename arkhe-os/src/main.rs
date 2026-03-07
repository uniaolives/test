//! Interface de linha de comando para o arkhe-os.
//! Permite criar tarefas e executar ciclos de escalonamento.

mod kernel;
mod intention;
mod net;
mod phys;
mod sensors;
mod telemetry;
mod anchor;
mod quantum;
mod lmt;
mod maestro;
mod security;
mod physics;
mod db;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod maestro_tests;

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use clap::Parser;
use std::io::{self, Write};
use futures::StreamExt;
use libp2p::{gossipsub, identity, swarm::SwarmEvent};
use chrono::Utc;
use std::collections::BTreeMap;
use tokio::time::{sleep, Duration};

use kernel::syscall::{SyscallHandler, SyscallResult};
use intention::parser::parse_intention_block;
use db::ledger::{TeknetLedger, HandoverRecord};
use arkhe_db::schema::{FutureCommitment, CommitmentStatus};
use phys::ibm_client::QuantumAntenna;
use sensors::ZPFEvent;
use crate::quantum::berry::{TopologicalQubit};
use telemetry::{BioEvent, GlobalState};
use net::stack::NetEvent;
use lmt::field::MeaningField;
use maestro::{PTPApiWrapper, MaestroSpine, MaestroOrchestrator, BranchingEngine};
use maestro::spine::PsiState;
use security::{XenoFirewall, XenoRiskLevel};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// RF bands to monitor (e.g. 2.4GHz,3.5GHz)
    #[arg(short, long, default_value = "2.4GHz")]
    bands: String,

    /// Miller Limit for Wave-Cloud nucleation
    #[arg(short, long, default_value_t = 4.64)]
    miller: f64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    env_logger::init();

    println!("======================================================");
    println!(" 🜁 ArkheOS Node v1.0 — LMT Integrated Foundation");
    println!(" PI DAY (3-14-2026) SINGULARITY TEST ENABLED");
    println!(" Bands: {} | Miller Limit: {}", args.bands, args.miller);
    println!(" [!] Timechain (RocksDB) conectada.");
    println!(" [!] Antena Horizontal (P2P) e Vertical (QPU) ativas.");
    println!(" Digite sua intenção em intention-lang ou 'exit' para sair.");
    println!("======================================================\n");

    // 1. Inicializa ou restaura o banco de dados
    let ledger = Arc::new(TeknetLedger::new("./arkhe_chain_db").expect("Erro fatal: DB não pôde ser montado."));
    let (restored_phi_q, restored_last_id) = ledger.restore_vacuum_state();

    let sys = Arc::new(Mutex::new(SyscallHandler::new(restored_phi_q)));
    let mut task_id_counter = restored_last_id + 1;

    // Maestro Hyper-Spine and Branching Logic
    let api_wrapper = PTPApiWrapper::new(Default::default());
    let maestro_spine = MaestroSpine::new(api_wrapper, "http://localhost:11434/v1/chat/completions");
    let branching_engine = Arc::new(RwLock::new(BranchingEngine::new(0.01)));
    let orchestrator = Arc::new(MaestroOrchestrator::new(maestro_spine, branching_engine));

    let state = Arc::new(GlobalState {
        phi_q: RwLock::new(restored_phi_q),
        coherence_history: RwLock::new(vec![]),
    });

    let (zpf_tx, mut zpf_rx) = mpsc::channel::<ZPFEvent>(1000);
    let (bio_tx, mut bio_rx) = mpsc::channel::<BioEvent>(1000);
    let (net_tx, mut net_rx) = mpsc::channel::<NetEvent>(1000);

    // Topological Hardware Simulation
    let _top_qubit = Arc::new(Mutex::new(TopologicalQubit::new()));
    let event_buffer: Arc<Mutex<BTreeMap<u64, String>>> = Arc::new(Mutex::new(BTreeMap::new()));

    // 2. Setup P2P
    let local_key = identity::Keypair::generate_ed25519();
    let mut swarm = libp2p::SwarmBuilder::with_existing_identity(local_key)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_behaviour(|key| {
            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .build()
                .map_err(|msg| io::Error::new(io::ErrorKind::Other, msg))?;
            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            ).map_err(|msg| io::Error::new(io::ErrorKind::Other, msg))?;
            let mdns = libp2p::mdns::tokio::Behaviour::new(libp2p::mdns::Config::default(), key.public().to_peer_id())?;
            Ok(net::ArkheNetBehavior { gossipsub, mdns })
        })?
        .build();

    let topic = gossipsub::IdentTopic::new("teknet/phi_q");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    // 3. Start Sensory Pipelines
    sensors::start_zpf_pipeline(zpf_tx).await;
    telemetry::start_bio_server(bio_tx, state.clone()).await;
    net::stack::start_multimodal_stack(net_tx, state.clone()).await;

    // 4. Fusion Engine
    let event_buffer_fusion = event_buffer.clone();
    tokio::spawn(async move {
        let mut analytics = sensors::analytics::MultivariateAnalytics::new(100);
        let mut entropy_engine = sensors::entropy::TransferEntropy::new(100);
        let mut meaning_field = MeaningField::new();

        loop {
            tokio::select! {
                Some(event) = zpf_rx.recv() => {
                    match event {
                        ZPFEvent::Spectrum { timestamp, kurtosis, .. } => {
                            let mut eb = event_buffer_fusion.lock().await;
                            eb.insert(timestamp, format!("ZPF_KURT: {:.2}", kurtosis));
                        },
                        ZPFEvent::MultiBand { timestamp_ns, bands } => {
                            for (band, power) in bands {
                                analytics.push(&band, power);
                            }
                            let m_kurt = analytics.mardia_kurtosis();
                            let mut eb = event_buffer_fusion.lock().await;
                            eb.insert(timestamp_ns, format!("MARDIA_KURT: {:.3}", m_kurt));

                            meaning_field.somatic = m_kurt.abs().min(1.0);
                            let resonance = meaning_field.calculate_resonance();
                            if resonance > 0.8 {
                                println!("\n[LMT] TRUTH PULSE: Resonance {:.3} at {}", resonance, timestamp_ns);
                            }
                        }
                    }
                }
                Some(event) = bio_rx.recv() => {
                    match event {
                        BioEvent::Telemetry { timestamp, accel, .. } => {
                            entropy_engine.push(accel, 1.0);
                            let te = entropy_engine.calculate();
                            let mut eb = event_buffer_fusion.lock().await;
                            eb.insert(timestamp, format!("TE_BIO: {:.4}", te));
                        }
                    }
                }
                Some(event) = net_rx.recv() => {
                    match event {
                        NetEvent::Update { timestamp, rssi, .. } => {
                            let mut eb = event_buffer_fusion.lock().await;
                            eb.insert(timestamp, format!("RSSI: {:.1}", rssi));
                        }
                    }
                }
            }
        }
    });

    // 5. Background Physics Recalibration
    let antenna = QuantumAntenna::new("SIMULATED_TOKEN".to_string());
    let state_phys = state.clone();
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(30)).await;
            if let Ok(real_phi_q) = antenna.measure_vacuum_quality("ibm_brisbane").await {
                let mut phi = state_phys.phi_q.write().await;
                *phi = real_phi_q;
                println!("\n[SYSTEM] Vácuo recalibrado via hardware físico: φ_q = {:.3}", real_phi_q);
            }
        }
    });

    // 6. Network Loop
    tokio::spawn(async move {
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::Behaviour(net::ArkheNetBehaviorEvent::Gossipsub(gossipsub::Event::Message { message, .. })) => {
                    println!("\n[NET] Gossip: {:?}", message.data);
                }
                SwarmEvent::Behaviour(net::ArkheNetBehaviorEvent::Mdns(libp2p::mdns::Event::Discovered(list))) => {
                    for (peer, _addr) in list {
                        println!("\n[NET] Peer Discovered: {}", peer);
                    }
                }
                _ => {}
            }
        }
    });

    // 7. REPL Loop
    loop {
        print!("arkhe> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() { continue; }
        if input == "exit" || input == "quit" { break; }

        if input == "status" {
            let phi = *state.phi_q.read().await;
            println!("[SYS] φ_q = {:.3} | Status: {}", phi, if phi > args.miller { "WAVE-CLOUD" } else { "STOCHASTIC" });
            continue;
        }

        if input.starts_with("intent ") {
            let intent_text = input[7..].to_string();
            let orchestrator_clone = orchestrator.clone();
            let psi_state = PsiState {
                current_coherence: *state.phi_q.read().await,
                ..Default::default()
            };

            tokio::spawn(async move {
                match orchestrator_clone.process_intent(&intent_text, &psi_state).await {
                    Ok(resp) => {
                        let risk = XenoFirewall::assess_risk(&resp, &psi_state);
                        if risk == XenoRiskLevel::Critical {
                            println!("\n[XENO-FIREWALL] ⚠ CRITICAL RISK DETECTED. CONTAINMENT ACTIVE.");
                        } else {
                            println!("\n[MAESTRO] Response: {}", resp);
                        }
                    },
                    Err(e) => println!("\n[MAESTRO] Error: {}", e),
                }
            });
            continue;
        }

        if input == "pi_handover" {
            println!("[FUTURE] Creating special handover to Ω(2030)...");
            let _commitment = FutureCommitment {
                id: "SINGULARITY_ACHIEVED_AT_π_DAY".to_string(),
                created_at: Utc::now(),
                target_at: Utc::now() + chrono::Duration::days(365 * 4),
                prediction_hash: "berry_phase_target_pi_2".to_string(),
                validation_signature: None,
                status: CommitmentStatus::Pending,
            };
            println!("[FUTURE] Commitment valid after 2030-03-14.");
            continue;
        }

        // Fallback to Intention Lang parser
        match parse_intention_block(input) {
            Ok((_, ast)) => {
                println!("  [+] Intenção Compilada: {}", ast.name);
                let mut sys_lock = sys.lock().await;
                sys_lock.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);

                if let SyscallResult::Success(msg) = sys_lock.sys_tick() {
                    println!("  [OK] {}", msg);
                    let phi = *state.phi_q.read().await;

                    let record = HandoverRecord {
                        id: task_id_counter,
                        timestamp: Utc::now(),
                        intention_name: ast.name.clone(),
                        coherence_delta: ast.coherence,
                        phi_q_after: phi,
                    };
                    ledger.commit_handover(&record).unwrap();
                    ledger.save_vacuum_state(phi, task_id_counter).unwrap();

                    task_id_counter += 1;
                }
            }
            Err(_) => println!("  [ERR] Comando desconhecido ou erro de sintaxe."),
        }
    }

    Ok(())
}
