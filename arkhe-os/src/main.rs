mod kernel;
mod lib;
mod intention;
mod net;
mod phys;
mod sensors;
mod telemetry;
mod anchor;
mod lmt;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use std::io::{self, Write};
use futures::StreamExt;
use libp2p::{gossipsub, identity, swarm::SwarmEvent};
use warp::Filter;
use chrono::Utc;
use std::collections::BTreeMap;

use kernel::syscall::{SyscallHandler, SyscallResult};
use intention::parser::parse_intention_block;
use arkhe_db::ledger::TeknetLedger;
use arkhe_db::schema::{Handover, HandoverStatus, FutureCommitment, CommitmentStatus};
use phys::ibm_sensor::IBMQuantumBridge;
use sensors::ZPFEvent;
use telemetry::{BioEvent, GlobalState};
use net::stack::NetEvent;
use lmt::field::MeaningField;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("======================================================");
    println!(" 🜁 ArkheOS Node v1.0 — LMT Integrated Foundation");
    println!(" [L] LITE integrated with [D] DATA");
    println!("======================================================\n");

    let ledger = Arc::new(TeknetLedger::new("arkhe_chain.log")?);
    let sys = Arc::new(Mutex::new(SyscallHandler::new(100.0)));
    let phys_bridge = Arc::new(IBMQuantumBridge::new("SIMULATED_TOKEN".to_string()));

    let state = Arc::new(GlobalState {
        phi_q: RwLock::new(1.0),
        coherence_history: RwLock::new(vec![]),
    });

    let (zpf_tx, mut zpf_rx) = mpsc::channel::<ZPFEvent>(1000);
    let (bio_tx, mut bio_rx) = mpsc::channel::<BioEvent>(1000);
    let (net_tx, mut net_rx) = mpsc::channel::<NetEvent>(1000);

    let event_buffer: Arc<Mutex<BTreeMap<u64, String>>> = Arc::new(Mutex::new(BTreeMap::new()));

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

    let zpf_tx_sync = zpf_tx.clone();
    tokio::spawn(async move {
        let socket = tokio::net::UdpSocket::bind("127.0.0.1:7001").await.unwrap();
        let mut buf = [0u8; 16];
        println!("[PTP] High-Precision UDP Sync active on 7001.");
        loop {
            if let Ok((len, _)) = socket.recv_from(&mut buf).await {
                if len == 16 {
                    let ts_ns = u64::from_le_bytes(buf[0..8].try_into().unwrap());
                    let val = f64::from_le_bytes(buf[8..16].try_into().unwrap());
                    let mut bands = std::collections::HashMap::new();
                    bands.insert("wifi".to_string(), val);
                    let _ = zpf_tx_sync.send(ZPFEvent::MultiBand { timestamp_ns: ts_ns, bands }).await;
                }
            }
        }
    });

    let event_buffer_fusion = event_buffer.clone();
    let _state_fusion = state.clone();
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

    let state_phys = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            if let Ok(real_phi) = phys_bridge.measure_physical_phi_q().await {
                let mut phi_q = state_phys.phi_q.write().await;
                *phi_q = real_phi;
            }
        }
    });

    tokio::spawn(async move {
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::Behaviour(net::ArkheNetBehaviorEvent::Gossipsub(gossipsub::Event::Message { message, .. })) => {
                    println!("\n[NET] Gossip: {:?}", message.data);
                }
                _ => {}
            }
        }
    });

    let state_mobile = state.clone();
    let mobile_route = warp::path("anchor")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let _state_inner = state_mobile.clone();
            ws.on_upgrade(move |mut websocket| async move {
                println!("[LMT] 👂 Smartphone Âncora Conectada!");
                while let Some(Ok(msg)) = websocket.next().await {
                    if let Ok(text) = msg.to_str() {
                        if let Ok(_data) = serde_json::from_str::<serde_json::Value>(text) {
                            // handle somatic signals
                        }
                    }
                }
            })
        });
    tokio::spawn(async move {
        warp::serve(mobile_route).run(([0, 0, 0, 0], 3030)).await;
    });

    loop {
        print!("arkhe> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() { continue; }
        if input == "exit" || input == "quit" { break; }

        let mut sys_lock = sys.lock().await;

        if input == "status" {
            let phi = *state.phi_q.read().await;
            println!("[SYS] φ_q = {:.3} | Status: {}", phi, if phi > 4.64 { "WAVE-CLOUD" } else { "STOCHASTIC" });
            continue;
        }

        if input.starts_with("commit ") {
            println!("[FUTURE] Recorded commitment for 2030.");
            continue;
        }

        match parse_intention_block(input) {
            Ok((_, ast)) => {
                sys_lock.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);
                if let SyscallResult::Success(msg) = sys_lock.sys_tick() {
                    println!("  [OK] {}", msg);
                    let phi = *state.phi_q.read().await;
                    let handover = Handover {
                        id: 0,
                        timestamp: Utc::now(),
                        source_epoch: 2026,
                        target_epoch: 2009,
                        description: format!("{} -> {}", ast.name, ast.target),
                        phi_q_before: phi,
                        phi_q_after: phi,
                        quantum_interest: 0.0,
                        status: HandoverStatus::Accepted,
                    };
                    ledger.commit_handover(handover)?;
                }
            }
            Err(_) => println!("  [ERR] Syntax Error."),
        }
    }

    Ok(())
}
