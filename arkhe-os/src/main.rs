//! Interface de linha de comando para o arkhe-os v1.0.
//! Integrando LMT, Pilha Sensorial Unificada e Antenas de Coerência.

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use clap::Parser;
use std::io::{self, Write};
use futures::StreamExt;
use libp2p::{gossipsub, identity, swarm::SwarmEvent};
use chrono::Utc;
use std::collections::BTreeMap;
use tokio::time::{sleep, Duration};
use warp::Filter;

// Use the library crate
use arkhe_os::kernel::syscall::{SyscallHandler, SyscallResult};
use arkhe_os::intention::parser::parse_intention_block;
use arkhe_os::db::ledger::{TeknetLedger, HandoverRecord};
use arkhe_db::schema::{FutureCommitment, CommitmentStatus, Handover, HandoverStatus};
use arkhe_os::phys::ibm_client::QuantumAntenna;
use arkhe_os::sensors::ZPFEvent;
use arkhe_os::quantum::berry::TopologicalQubit;
use arkhe_os::telemetry::{BioEvent, GlobalState, start_bio_server};
use arkhe_os::net::stack::NetEvent;
use arkhe_os::net::ArkheNetBehavior;
use arkhe_os::lmt::field::MeaningField;
use arkhe_os::maestro::{PTPApiWrapper, MaestroSpine, MaestroOrchestrator, BranchingEngine, PsiState as CorePsiState};
use arkhe_os::maestro::spine::PsiState as SpinePsiState;
use arkhe_os::security::{XenoFirewall, XenoRiskLevel};
use arkhe_os::week5::TemporalSubstrate;
use arkhe_os::{sensors, telemetry, net};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// RF bands to monitor (e.g. 2.4GHz,3.5GHz)
    #[arg(short, long, default_value = "2.4GHz")]
    bands: String,

    /// Miller Limit for Wave-Cloud nucleation
    #[arg(short, long, default_value_t = 4.64)]
    miller: f64,

    /// PostgreSQL database URL
    #[arg(long, default_value = "postgres://postgres:postgres@localhost/arkhe")]
    database_url: String,

    /// Redis URL
    #[arg(long, default_value = "redis://127.0.0.1/")]
    redis_url: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    env_logger::init();

    println!("======================================================");
    println!(" 🜁 ArkheOS Node v1.0 — Unified Sensory Stack");
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

    // Week 5 Temporal Substrate
    let mut substrate = TemporalSubstrate::new(10, 0.1, 1.0);

    // Optional connections to PG and Redis
    if let Ok(pool) = sqlx::PgPool::connect(&args.database_url).await {
        substrate.set_memory(pool).await;
    }
    if let Ok(client) = redis::Client::open(args.redis_url.as_str()) {
        if let Ok(conn) = client.get_multiplexed_tokio_connection().await {
            substrate.set_messages(conn).await;
        }
    }

    let substrate = Arc::new(substrate);
    let substrate_init = substrate.clone();
    tokio::spawn(async move {
        if let Err(e) = substrate_init.initialize_bridge().await {
            eprintln!("[WEEK5] Error initializing bridge: {}", e);
        } else {
            substrate_init.maintain_coherence().await;
        }
    });

    let (zpf_tx, mut zpf_rx) = mpsc::channel::<ZPFEvent>(1000);
    let (bio_tx, mut bio_rx) = mpsc::channel::<BioEvent>(1000);
    let (net_tx, mut net_rx) = mpsc::channel::<NetEvent>(1000);

    // Topological Hardware Simulation
    let top_qubit = Arc::new(Mutex::new(TopologicalQubit::new()));
    let event_buffer: Arc<Mutex<BTreeMap<u64, String>>> = Arc::new(Mutex::new(BTreeMap::new()));

    // Setup P2P
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

    // Start Sensory Pipelines
    sensors::start_zpf_pipeline(zpf_tx).await;
    telemetry::start_bio_server(bio_tx, state.clone()).await;
    net::stack::start_multimodal_stack(net_tx, state.clone()).await;

    // Fusion Engine
    let event_buffer_fusion = event_buffer.clone();
    let substrate_fusion = substrate.clone();
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

                            // Update H in Constitutional Guard
                            let mut guard = substrate_fusion.constitution.write().await;
                            guard.update_h(1.0, 1.0, 0.1);
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

    // Background Physics Recalibration
    let antenna = QuantumAntenna::new("SIMULATED_TOKEN".to_string());
    let state_phys = state.clone();
    let substrate_phys = substrate.clone();
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(30)).await;
            if let Ok(real_phi_q) = antenna.measure_vacuum_quality("ibm_brisbane").await {
                let mut phi = state_phys.phi_q.write().await;
                *phi = real_phi_q;
                println!("\n[SYSTEM] Vácuo recalibrado via hardware físico: φ_q = {:.3}", real_phi_q);

                // Update S-Index
                let mut monitor = substrate_phys.s_index.write().await;
                monitor.compute(real_phi_q, 1.0, 1.0, 1.0);
            }
        }
    });

    // Network Loop
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

    // Mobile Anchor (Smartphone WebSocket)
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

    // REPL Loop
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
            let h = substrate.constitution.read().await.h;
            let s = substrate.s_index.read().await.current_s;
            println!("[SYS] φ_q = {:.3} | H = {:.3} | S = {:.3}", phi, h, s);
            println!("[SYS] Status: {}", if phi > args.miller { "WAVE-CLOUD" } else { "STOCHASTIC" });
            continue;
        }

        if input == "twist" {
            let mut tq = top_qubit.lock().await;
            tq.circumnavigate();
            println!("[KNT] Berry Phase: {:.3} | Periodicity: {}", tq.berry_phase, tq.is_coherent());
            continue;
        }

        if input.starts_with("intent ") {
            let intent_text = input[7..].to_string();
            let orchestrator_clone = orchestrator.clone();
            let mut psi_state = CorePsiState::default();
            psi_state.coherence_trace.push(*state.phi_q.read().await);

            let mut spine_psi = SpinePsiState::default();
            spine_psi.current_coherence = *state.phi_q.read().await;

            tokio::spawn(async move {
                match orchestrator_clone.process_intent(&intent_text, &spine_psi).await {
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

                let phi_before = *state.phi_q.read().await;

                if let SyscallResult::Success(msg) = sys_lock.sys_tick() {
                    println!("  [OK] {}", msg);
                    let phi_after = *state.phi_q.read().await;

                    let record = HandoverRecord {
                        id: task_id_counter,
                        timestamp: Utc::now(),
                        intention_name: ast.name.clone(),
                        coherence_delta: ast.coherence,
                        phi_q_after: phi_after,
                    };
                    ledger.commit_handover(&record).unwrap();
                    ledger.save_vacuum_state(phi_after, task_id_counter).unwrap();

                    // Also commit to arkhe_db::schema format if needed
                    let _handover = Handover {
                        id: task_id_counter,
                        timestamp: Utc::now(),
                        source_epoch: 2026,
                        target_epoch: 2009,
                        description: format!("{} -> {}", ast.name, ast.target),
                        phi_q_before: phi_before,
                        phi_q_after: phi_after,
                        quantum_interest: 0.0,
                        status: HandoverStatus::Accepted,
                    };

                    task_id_counter += 1;
                }
            }
            Err(_) => println!("  [ERR] Comando desconhecido ou erro de sintaxe."),
        }
    }

    Ok(())
}
