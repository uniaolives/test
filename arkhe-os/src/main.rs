mod kernel;
mod lib;
mod intention;
mod net;
mod phys;
mod sensors;
mod telemetry;
mod anchor;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use std::io::{self, Write};
use futures::StreamExt;
use libp2p::{gossipsub, identity, swarm::SwarmEvent};
use warp::Filter;
use chrono::Utc;

use kernel::syscall::{SyscallHandler, SyscallResult};
use intention::parser::parse_intention_block;
use arkhe_db::ledger::TeknetLedger;
use arkhe_db::schema::{Handover, HandoverStatus, FutureCommitment, CommitmentStatus};
use phys::ibm_sensor::IBMQuantumBridge;
use sensors::ZPFEvent;
use telemetry::{BioEvent, GlobalState};
use net::stack::NetEvent;
use std::collections::BTreeMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("======================================================");
    println!(" 🜁 ArkheOS Node v1.0 — Unified Sensory Stack");
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

    // BTreeMap for high-precision event ordering (PTP-simulated)
    let event_buffer: Arc<Mutex<BTreeMap<u64, String>>> = Arc::new(Mutex::new(BTreeMap::new()));

    // 1. Setup P2P
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

    // 2. Start Sensory Pipelines
    sensors::start_zpf_pipeline(zpf_tx).await;
    telemetry::start_bio_server(bio_tx, state.clone()).await;
    net::stack::start_multimodal_stack(net_tx, state.clone()).await;

    // 3. Fusion Engine (Background)
    let event_buffer_fusion = event_buffer.clone();
    tokio::spawn(async move {
        let mut kurtosis_engine = sensors::analytics::MultivariateKurtosis::new(100);
        let mut entropy_engine = sensors::entropy::TransferEntropy::new(100);
        loop {
            tokio::select! {
                Some(event) = zpf_rx.recv() => {
                    match event {
                        ZPFEvent::Spectrum { timestamp, kurtosis, .. } => {
                            let mut eb = event_buffer_fusion.lock().await;
                            eb.insert(timestamp, format!("ZPF_ANOMALY: {:.2}", kurtosis));
                            if kurtosis > 2.0 {
                                println!("\n[FUSION] SDR Anomaly Detected (Kurtosis: {:.2}) at {}", kurtosis, timestamp);
                            }
                        },
                        ZPFEvent::MultiBand { timestamp_ns, bands } => {
                            for (band, power) in bands {
                                kurtosis_engine.push(&band, power);
                            }
                            let m_kurt = kurtosis_engine.calculate();
                            let mut eb = event_buffer_fusion.lock().await;
                            eb.insert(timestamp_ns, format!("MULTI_KURT: {:.3}", m_kurt));
                        }
                    }
                }
                Some(event) = bio_rx.recv() => {
                    match event {
                        BioEvent::Telemetry { timestamp, accel, .. } => {
                            entropy_engine.push(accel, 1.0); // Simulate Y=1.0 for now
                            let te = entropy_engine.calculate();
                            let mut eb = event_buffer_fusion.lock().await;
                            eb.insert(timestamp, format!("TE_BIO_PLENUM: {:.4}", te));

                            if accel > 2.0 {
                                println!("\n[FUSION] Bio-Turbulence Detected (Accel: {:.2})", accel);
                            }
                        }
                    }
                }
                Some(event) = net_rx.recv() => {
                    match event {
                        NetEvent::Update { rssi, .. } => {
                            if rssi < -80.0 {
                                println!("\n[FUSION] Network Coherence Low (RSSI: {:.1})", rssi);
                            }
                        }
                    }
                }
            }
        }
    });

    // 4. Background Physics recalibration
    let state_phys = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            if let Ok(real_phi) = phys_bridge.measure_physical_phi_q().await {
                println!("\n[PHYSICS] IBM Quantum Pulse: φ_q = {:.3}", real_phi);
                let mut phi_q = state_phys.phi_q.write().await;
                *phi_q = real_phi;
            }
        }
    });

    // 5. Network Loop
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

    // 6. Shell
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
            match sys_lock.sys_coherence_status() {
                SyscallResult::CoherenceUpdate(avail) => {
                    match sys_lock.sys_check_nucleation() {
                        SyscallResult::WaveCloudStatus(n, phi) => {
                            println!("[SYS] φ_q = {:.3} | Coherence: {:.3} | Wave-Cloud: {}", phi, avail, n);
                        }
                        _ => ()
                    }
                }
                _ => ()
            }
            continue;
        }

        if input.starts_with("commit ") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() >= 3 {
                let id = parts[1];
                let hash = parts[2];
                let _commitment = FutureCommitment {
                    id: id.to_string(),
                    created_at: Utc::now(),
                    target_at: Utc::now() + chrono::Duration::days(365 * 4), // 2030 target
                    prediction_hash: hash.to_string(),
                    validation_signature: None,
                    status: CommitmentStatus::Pending,
                };
                println!("[FUTURE] Recorded commitment for 2030: {}", id);
                // In production, we would record this in the ledger/engine
            }
            continue;
        }

        match parse_intention_block(input) {
            Ok((_, ast)) => {
                sys_lock.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);
                if let SyscallResult::Success(msg) = sys_lock.sys_tick() {
                    println!("  [OK] {}", msg);
                    let phi = match sys_lock.sys_check_nucleation() {
                        SyscallResult::WaveCloudStatus(_, p) => p,
                        _ => 0.0
                    };
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
