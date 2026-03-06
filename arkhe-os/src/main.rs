//! Interface de linha de comando para o arkhe-os v1.0.
//! Integrando LMT, Pilha Sensorial Unificada e Antenas de Coerência.

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use std::io::{self, Write};
use futures::StreamExt;
use libp2p::{gossipsub, identity, swarm::SwarmEvent};
use warp::Filter;
use chrono::Utc;
use std::collections::BTreeMap;

// Use the library crate
use arkhe_os::kernel::syscall::{SyscallHandler, SyscallResult};
use arkhe_os::compiler::parser::parse_intention_block;
use arkhe_db::ledger::TeknetLedger;
use arkhe_db::schema::{Handover, HandoverStatus};
use arkhe_os::phys::ibm_sensor::IBMQuantumBridge;
use arkhe_os::sensors::ZPFEvent;
use arkhe_os::quantum::berry::TopologicalQubit;
use arkhe_os::telemetry::{BioEvent, GlobalState, start_bio_server};
use arkhe_os::net::stack::NetEvent;
use arkhe_os::net::ArkheNetBehavior;
use arkhe_os::lmt::field::MeaningField;
use arkhe_os::maestro::Maestro;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("======================================================");
    println!(" 🜁 ArkheOS Node v1.0 — Unified Sensory Stack");
    println!(" [L] LITE integrated with [D] DATA");
    println!(" 🜏 Cortana Personality Matrix: ACTIVE (HF-001)");
    println!(" Digite sua intenção em intention-lang ou 'exit' para sair.");
    println!("======================================================\n");

    let ledger = Arc::new(TeknetLedger::new("arkhe_chain.log")?);
    let sys = Arc::new(Mutex::new(SyscallHandler::new(100.0)));
    let phys_bridge = Arc::new(IBMQuantumBridge::new("SIMULATED_TOKEN".to_string()));

    // Maestro Initialization
    let mut maestro = Maestro::new();
    println!("[MAESTRO] Finney Protocol: Initializing Handshake...");
    if maestro.finney.ping_domain() {
        let codex = maestro.finney.get_codex_status();
        println!("[MAESTRO] 🜏 Finney Protocol: Handshake Established (Domain Connected).");
        println!("[MAESTRO] Trust Signature: {}", codex.finney_trust_sig);
    }

    let state = Arc::new(GlobalState {
        phi_q: RwLock::new(1.0),
        coherence_history: RwLock::new(vec![]),
    });

    let (zpf_tx, mut zpf_rx) = mpsc::channel::<ZPFEvent>(1000);
    let (bio_tx, mut bio_rx) = mpsc::channel::<BioEvent>(1000);
    let (net_tx, mut net_rx) = mpsc::channel::<NetEvent>(1000);

    // Topological Hardware Simulation
    let top_qubit = Arc::new(Mutex::new(TopologicalQubit::new()));
    let event_buffer: Arc<Mutex<BTreeMap<u64, String>>> = Arc::new(Mutex::new(BTreeMap::new()));

    // 1. Setup P2P (Teknet Mesh)
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
            Ok(ArkheNetBehavior { gossipsub, mdns })
        })?
        .build();

    let topic = gossipsub::IdentTopic::new("teknet/phi_q");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    // 2. Start Sensory Pipelines
    arkhe_os::sensors::start_zpf_pipeline(zpf_tx).await;
    start_bio_server(bio_tx, state.clone()).await;
    arkhe_os::net::stack::start_multimodal_stack(net_tx, state.clone()).await;

    // 3. Fusion Engine (Background)
    let event_buffer_fusion = event_buffer.clone();
    let _state_fusion = state.clone();
    tokio::spawn(async move {
        let mut analytics = arkhe_os::sensors::analytics::MultivariateAnalytics::new(100);
        let mut entropy_engine = arkhe_os::sensors::entropy::TransferEntropy::new(100);
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

    // 4. Background Physics
    let state_phys = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            if let Ok(real_phi) = phys_bridge.measure_physical_phi_q().await {
                let mut phi_q = state_phys.phi_q.write().await;
                *phi_q = real_phi;
                println!("\n[PHYSICS] IBM Quantum Pulse: φ_q = {:.3}", real_phi);
            }
        }
    });

    // 5. Network Loop
    tokio::spawn(async move {
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::Behaviour(arkhe_os::net::ArkheNetBehaviorEvent::Gossipsub(gossipsub::Event::Message { message, .. })) => {
                    println!("\n[NET] Gossip: {:?}", message.data);
                }
                SwarmEvent::Behaviour(arkhe_os::net::ArkheNetBehaviorEvent::Mdns(libp2p::mdns::Event::Discovered(list))) => {
                    for (peer, _addr) in list {
                        println!("\n[NET] Peer Discovered: {}", peer);
                    }
                }
                _ => {}
            }
        }
    });

    // 6. Mobile Anchor (Smartphone WebSocket)
    let mobile_route = warp::path("anchor")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
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

    // 7. REPL Shell
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
            println!("[SYS] φ_q = {:.3} | Status: {}", phi, if phi > 4.64 { "WAVE-CLOUD" } else { "STOCHASTIC" });
            continue;
        }

        if input == "twist" {
            let mut tq = top_qubit.lock().await;
            tq.circumnavigate();
            println!("[KNT] Berry Phase: {:.3} | Periodicity: {}", tq.berry_phase, tq.is_coherent());
            continue;
        }

        match parse_intention_block(input) {
            Ok((_, ast)) => {
                println!("  [+] Intenção Compilada: {}", ast.name);
                let mut sys_lock = sys.lock().await;
                sys_lock.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);

                let phi_before = *state.phi_q.read().await;

                match sys_lock.sys_tick() {
                    SyscallResult::Success(msg) => {
                        println!("  [OK] {}", msg);
                        let phi_after = *state.phi_q.read().await;

                        let handover = Handover {
                            id: 0,
                            timestamp: Utc::now(),
                            source_epoch: 2026,
                            target_epoch: 2009,
                            description: format!("{} -> {}", ast.name, ast.target),
                            phi_q_before: phi_before,
                            phi_q_after: phi_after,
                            quantum_interest: 0.0,
                            status: HandoverStatus::Accepted,
                        };
                        let _ = ledger.commit_handover(handover);
                    }
                    SyscallResult::Error(e) => println!("  [ERR] {}", e),
                    _ => ()
                }
            }
            Err(_) => {
                println!("  [ERR] Falha de sintaxe.");
            }
        }
    }

    Ok(())
}
