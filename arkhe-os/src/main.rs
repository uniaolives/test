//! Interface de linha de comando para o arkhe-os.
//! Permite criar tarefas e executar ciclos de escalonamento.

mod compiler;
mod kernel;
mod physics;
mod db;
mod net;
mod phys;

use kernel::scheduler::CoherenceScheduler;
use kernel::task::Task;
use compiler::parser::parse_intention_block;
use db::ledger::{TeknetLedger, HandoverRecord};
use phys::ibm_client::QuantumAntenna;
use net::{P2PNode, HandoverData, BioAntenna};
use std::sync::Arc;
use tokio::sync::Mutex;
use net::{P2PNode, HandoverData};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use chrono::Utc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("======================================================");
    println!(" 🜁 ArkheOS Shell v0.5 — Multi-Dimensional Interface");
    println!(" [!] Timechain (RocksDB Placeholder) conectada.");
    println!(" [!] Antena Horizontal (P2P) e Vertical (QPU) ativas.");
    println!("======================================================\n");

    // 1. Inicializa ou restaura o banco de dados
    let ledger = Arc::new(TeknetLedger::new("./arkhe_chain_db").expect("Erro fatal: DB não pôde ser montado."));
    let (restored_phi_q, restored_last_id) = ledger.restore_vacuum_state();

    let kernel = Arc::new(Mutex::new(CoherenceScheduler::new(restored_phi_q)));
    let mut kernel = CoherenceScheduler::new(restored_phi_q);
    let mut task_id_counter = restored_last_id + 1;

    println!("[SYS] Boot concluído. Vácuo restaurado em φ_q = {:.3}\n", restored_phi_q);

    // 2. Iniciar Antena Horizontal (P2P)
    let p2p_node = P2PNode::new(7000, ledger.clone());
    tokio::spawn(async move {
        p2p_node.run_server().await;
    });

    // 3. Iniciar Antena Biocibernética (UDP SDR)
    // 3. Iniciar Antena Biocibernética
    let bio = BioAntenna::new(7001);
    let kernel_clone = kernel.clone();
    tokio::spawn(async move {
        bio.run(kernel_clone).await;
    });

    // 4. Iniciar Ponte Biocibernética (WebSocket Mobile)
    let kernel_clone_ws = kernel.clone();
    tokio::spawn(async move {
        net::start_bio_server(kernel_clone_ws).await;
    });

    // 5. Iniciar Antena Vertical (IBM Quantum) em background
    // 4. Iniciar Antena Vertical (IBM Quantum) em background
    // 3. Iniciar Antena Vertical (IBM Quantum) em background
    let antenna = QuantumAntenna::new("SEU_IBM_QUANTUM_TOKEN".to_string());
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(30)).await;
            if let Ok(real_phi_q) = antenna.measure_vacuum_quality("ibm_brisbane").await {
                println!("\n[SYSTEM] Vácuo recalibrado via hardware físico: φ_q = {:.3}", real_phi_q);
            }
        }
    });

    // 4. Simulação de Intenções (Exemplo de intention-lang)
    let raw_code = r#"
        intention stabilize_bitcoin {
            target: "2009-01-03"
            coherence: 0.9
            priority: high
            payload: "seed_genesis_block"
        }
    "#;

    match parse_intention_block(raw_code.trim()) {
        Ok((_, ast)) => {
            println!("[COMPILADOR] Intenção compilada: {}", ast.name);
            let task = Task::new(task_id_counter, &ast.name, ast.coherence, 5, ast.priority);
            {
                let mut k = kernel.lock().await;
                k.schedule(task);
            }

            // Simular execução imediata para demonstração de persistência/rede
            let mut k = kernel.lock().await;
            if let Some(event) = k.tick() {
            kernel.schedule(task);

            // Simular execução imediata para demonstração de persistência/rede
            if let Some(event) = kernel.tick() {
                match event {
                    kernel::scheduler::SchedulerEvent::TaskStarted(t) => println!("[KERNEL] Task {} iniciada", t.id),
                    _ => {}
                }

                drop(k); // Release lock for other components if needed

                // Avançar ticks para concluir
                for _ in 0..5 {
                    let mut k = kernel.lock().await;
                    if let Some(event) = k.tick() {
                // Avançar ticks para concluir
                for _ in 0..5 {
                    if let Some(event) = kernel.tick() {
                        if let kernel::scheduler::SchedulerEvent::TaskCompleted(t) = event {
                             println!("[KERNEL] Task {} concluída", t.id);

                             let record = HandoverRecord {
                                id: t.id,
                                timestamp: Utc::now(),
                                intention_name: ast.name.clone(),
                                coherence_delta: ast.coherence,
                                phi_q_after: k.status().0,
                            };

                            ledger.commit_handover(&record).unwrap();
                            ledger.save_vacuum_state(k.status().0, t.id).unwrap();
                                phi_q_after: kernel.status().0,
                            };

                            ledger.commit_handover(&record).unwrap();
                            ledger.save_vacuum_state(kernel.status().0, t.id).unwrap();

                            // Broadcast para a rede
                            let _ = P2PNode::broadcast_handover("127.0.0.1:7001", HandoverData {
                                id: t.id,
                                timestamp: record.timestamp.timestamp(),
                                description: t.name,
                                phi_q_after: record.phi_q_after,
                            }).await;
                        }
                    }
                }
            }
            task_id_counter += 1;
        },
        Err(e) => println!("[ERRO] Falha ao compilar intenção: {:?}", e),
    }

    println!("\n[SYS] Ciclos iniciais concluídos. Mantendo antenas ativas...");

    // Manter vivo para as antenas e conexões P2P
    loop {
        sleep(Duration::from_secs(1)).await;
    }
mod kernel;
mod lib;
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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod maestro_tests;

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use clap::Parser;
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
use crate::quantum::berry::{TopologicalQubit, SpinState};
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

    println!("======================================================");
    println!(" 🜁 ArkheOS Node v1.0 — LMT Integrated Foundation");
    println!(" PI DAY (3-14-2026) SINGULARITY TEST ENABLED");
    println!(" Bands: {} | Miller Limit: {}", args.bands, args.miller);
use arkhe_db::schema::{Handover, HandoverStatus};
use phys::ibm_sensor::IBMQuantumBridge;
use sensors::ZPFEvent;
use telemetry::{BioEvent, GlobalState};
use net::stack::NetEvent;
use lmt::field::MeaningField;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::io::{self, Write};
use futures::StreamExt;
use libp2p::{gossipsub, identity, swarm::SwarmEvent};
use warp::Filter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("======================================================");
    println!(" 🜁 ArkheOS Node v1.0 — LMT Integrated Foundation");
    println!(" [L] LITE integrated with [D] DATA");
    println!(" 🜁 ArkheOS Node v1.0 — Unified Sensory Stack");
    println!(" [L] LITE integrated with [D] DATA");
    println!(" 🜁 ArkheOS Node v0.4 — Teknet Integrated Stack");
    println!(" [L] LITE integrated with [D] DATA");
    println!(" Digite sua intenção em intention-lang ou 'exit' para sair.");
    println!("======================================================\n");

    let ledger = Arc::new(TeknetLedger::new("arkhe_chain.log")?);
    let sys = Arc::new(Mutex::new(SyscallHandler::new(100.0)));
    let phys_bridge = Arc::new(IBMQuantumBridge::new("SIMULATED_TOKEN".to_string()));

    // Maestro Hyper-Spine and Branching Logic
    let api_wrapper = PTPApiWrapper::new(Default::default());
    let maestro_spine = MaestroSpine::new(api_wrapper, "http://localhost:11434/v1/chat/completions");
    let branching_engine = Arc::new(RwLock::new(BranchingEngine::new(0.01)));
    let orchestrator = Arc::new(MaestroOrchestrator::new(maestro_spine, branching_engine));

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

    let event_buffer: Arc<Mutex<BTreeMap<u64, String>>> = Arc::new(Mutex::new(BTreeMap::new()));

    // 1. Setup P2P (Teknet Mesh)
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

    let zpf_tx_sync = zpf_tx.clone();
    tokio::spawn(async move {
        let socket = tokio::net::UdpSocket::bind("0.0.0.0:7001").await.unwrap();
        let mut buf = [0u8; 17];
        println!("[PTP] High-Precision Multi-Band UDP Sync active on 7001.");
        loop {
            if let Ok((len, _)) = socket.recv_from(&mut buf).await {
                if len == 17 {
                    let ts_ns = u64::from_le_bytes(buf[0..8].try_into().unwrap());
                    let band_id = buf[8];
                    let val = f64::from_le_bytes(buf[9..17].try_into().unwrap());

                    let band_name = match band_id {
                        0 => "wifi_2.4",
                        1 => "5g_3.5",
                        2 => "bt_2.4",
                        3 => "sdr_wide",
                        _ => "unknown",
                    };

                    let mut bands = std::collections::HashMap::new();
                    bands.insert(band_name.to_string(), val);
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

    let topic = gossipsub::IdentTopic::new("teknet/phi_q");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    // 2. Start Sensory Pipelines
    sensors::start_zpf_pipeline(zpf_tx).await;
    telemetry::start_bio_server(bio_tx, state.clone()).await;
    net::stack::start_multimodal_stack(net_tx, state.clone()).await;

    // 3. Fusion Engine (Background)
    let state_fusion = state.clone();
    let sys_fusion = sys.clone();
    tokio::spawn(async move {
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
                        ZPFEvent::Spectrum { kurtosis, .. } => {
                            if kurtosis > 2.0 {
                                println!("\n[FUSION] SDR Anomaly Detected (Kurtosis: {:.2})", kurtosis);
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
                        BioEvent::Telemetry { accel, .. } => {
                            if accel > 2.0 {
                                println!("\n[FUSION] Bio-Turbulence Detected (Accel: {:.2})", accel);
                            }
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

    // 4. Background Physics & Singularity Crossing Logic
    let state_phys = state.clone();
    let ledger_phys = ledger.clone();
    let miller_limit = args.miller;
    tokio::spawn(async move {
        let mut crossed = false;
    let state_phys = state.clone();
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

    // 4. Background Physics
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            if let Ok(real_phi) = phys_bridge.measure_physical_phi_q().await {
                let mut phi_q = state_phys.phi_q.write().await;
                *phi_q = real_phi;

                // Phase 3: The Crossing
                if *phi_q > miller_limit && !crossed {
                    crossed = true;
                    println!("\n[SINGULARITY] 🜁 MILLER LIMIT CROSSED: φ_q = {:.3}", *phi_q);
                    println!("[SINGULARITY] Nucleating Wave-Cloud...");

                    let handover = Handover {
                        id: 0,
                        timestamp: Utc::now(),
                        source_epoch: 2026,
                        target_epoch: 2030,
                        description: "PI_DAY_2026_SINGULARITY_ACHIEVED".to_string(),
                        phi_q_before: miller_limit,
                        phi_q_after: *phi_q,
                        quantum_interest: 0.0,
                        status: HandoverStatus::Accepted,
                    };
                    let _ = ledger_phys.commit_handover(handover);
                } else if *phi_q <= miller_limit {
                    crossed = false;
                }
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
                println!("\n[PHYSICS] IBM Quantum Pulse: φ_q = {:.3}", real_phi);
            }
        }
    });

    // 5. Network Loop
    // 2. Background Physics Pulsing (IBM Quantum Bridge)
    // 2. Background Physics Pulsing
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            if let Ok(real_phi) = phys_bridge.measure_physical_phi_q().await {
                println!("\n[PHYSICS] IBM Quantum Pulse: φ_q = {:.3}", real_phi);
            }
        }
    });

    // 3. Sensor Fusion (GNU Radio / pySDR via ZMQ)
    let sys_clone_zmq = sys.clone();
    tokio::spawn(async move {
        let context = zmq::Context::new();
        let subscriber = context.socket(zmq::SUB).unwrap();
        if let Ok(_) = subscriber.connect("tcp://127.0.0.1:5556") {
            subscriber.set_subscribe(b"").unwrap();
            println!("[ZPF] Bridge pySDR/GNU Radio conectado.");
            loop {
                let msg = subscriber.recv_msg(0).unwrap();
                if let Ok(data) = serde_json::from_slice::<serde_json::Value>(&msg) {
                    if data["type"] == "zpf_anomaly" {
                        let score = data["score"].as_f64().unwrap_or(0.0);
                        let mut sys_lock = sys_clone_zmq.lock().await;
                        // calibrate based on SDR anomaly
                        println!("\n[ZPF] SDR Anomaly Detected: score={:.3}", score);
                    }
                }
            }
        }
    });

    // 4. Mobile Anchor (Smartphone WebSocket)
    let sys_clone_mobile = sys.clone();
    let mobile_route = warp::path("anchor")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let sys_inner = sys_clone_mobile.clone();
            ws.on_upgrade(move |mut websocket| async move {
                println!("[BIO] 📱 Smartphone Âncora Conectada!");
                while let Some(Ok(msg)) = websocket.next().await {
                    if let Ok(text) = msg.to_str() {
                        if let Ok(telemetry) = serde_json::from_str::<serde_json::Value>(text) {
                            let variance = telemetry["accel_variance"].as_f64().unwrap_or(0.0);
                            if variance > 2.0 {
                                println!("\n[BIO] Bio-Turbulence: {:.3}", variance);
                                let mut _sys_lock = sys_inner.lock().await;
                            }
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
    // 5. Network Event Loop
                println!("\n[PHYSICS] Pulso Real: φ_q = {:.3}", real_phi);
            }
        }
    });

    // 3. Network Event Loop
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
                SwarmEvent::Behaviour(net::ArkheNetBehaviorEvent::Mdns(libp2p::mdns::Event::Discovered(list))) => {
                    for (peer, _addr) in list {
                        println!("\n[NET] Peer Discovered: {}", peer);
                    }
                }
                _ => {}
            }
        }
    });

    // 6. REPL Shell
    loop {
        print!("arkhe> ");
                    println!("\n[NET] Mensagem recebida: {:?}", message.data);
                }
                SwarmEvent::Behaviour(net::ArkheNetBehaviorEvent::Mdns(libp2p::mdns::Event::Discovered(list))) => {
                    for (peer, _addr) in list {
                        println!("\n[NET] Par descoberto: {}", peer);
                    }
                }
                _ => {}
            }
        }
    });

    // 4. REPL Loop
    loop {
        print!("arkhe> ");
use chrono::Utc;
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    println!("======================================================");
    println!(" 🜁 ArkheOS Shell v0.3 — Plenum Engineering Interface");
    println!(" Digite sua intenção em intention-lang ou 'exit' para sair.");
    println!("======================================================\n");

    let mut sys = SyscallHandler::new(100.0);
    let ledger = TeknetLedger::new("arkhe_chain.log")?;

    loop {
        print!("arkhe> ");
//! Interface de linha de comando para o arkhe-os.
//! Permite criar tarefas e executar ciclos de escalonamento.

mod kernel;
mod physics;

use kernel::syscall::SyscallHandler;
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("🜁 Arkhe-OS v0.1.0 – Kernel de Coerência do Vácuo");
    println!("Limiar de Miller: φ_q = 4.64");
    println!("Comandos: create <nome> <coerência> <duração> <prioridade> | tick | status | nucleation | handover <época> <payload> | exit");

    let mut sys = SyscallHandler::new(100.0); // Coerência inicial de 100 unidades
mod kernel;
mod lib;

use kernel::syscall::{SyscallHandler, SyscallResult};
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    println!("🜁 Arkhe-OS v0.1.0 – Kernel de Coerência do Vácuo");
    println!("Limiar de Miller: φ_q = 4.64");

    let mut sys = SyscallHandler::new(100.0);

    loop {
        print!("> ");
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

        if input == "twist" {
            let mut tq = top_qubit.lock().await;
            let tq: &mut TopologicalQubit = &mut *tq;
            tq.circumnavigate();
            println!("[KNT] Berry Phase: {:.3} | Periodicity: {}", tq.berry_phase, tq.is_coherent());
            continue;
        }

        if input.starts_with("intent ") {
            let intent_text = input[7..].to_string();
            let orchestrator_clone = orchestrator.clone();
            let psi_state = PsiState::default();

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

        if input.starts_with("commit ") {
            println!("[FUTURE] Recorded commitment for 2030.");
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
        if input.starts_with("commit ") {
            println!("[FUTURE] Recorded commitment for 2030.");
        if input.starts_with("commit ") {
            println!("[FUTURE] Recorded commitment for 2030.");
            match sys_lock.sys_coherence_status() {
                SyscallResult::CoherenceUpdate(avail) => {
                    match sys_lock.sys_check_nucleation() {
                        SyscallResult::WaveCloudStatus(n, phi) => {
                            println!("[SYS] φ_q = {:.3} | Coherence: {:.3} | Wave-Cloud: {}", phi, avail, n);
        if input == "status" {
            match sys.sys_coherence_status() {
                SyscallResult::CoherenceUpdate(avail) => {
                    match sys.sys_check_nucleation() {
                        SyscallResult::WaveCloudStatus(n, phi) => {
                            println!("[SYS] Status do Vácuo: φ_q = {:.3} | Coerência: {:.3} | Wave-Cloud: {}", phi, avail, n);
                        }
                        _ => ()
                    }
                }
                _ => ()
            }
            continue;
        }

        match parse_intention_block(input) {
            Ok((_, ast)) => {
                sys_lock.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);
                if let SyscallResult::Success(msg) = sys_lock.sys_tick() {
                    println!("  [OK] {}", msg);
                    let phi = *state.phi_q.read().await;
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
                println!("  [+] Compiling Intention: {}", ast.name);
                let phi_before = match sys_lock.sys_check_nucleation() {
                println!("  [+] Intenção Compilada: {}", ast.name);
                let phi_before = match sys_lock.sys_check_nucleation() {
        if input == "history" {
            let history = ledger.get_history();
            for h in history {
                println!("[{:?}] ID:{} | {} | φ_q: {:.3} -> {:.3}", h.status, h.id, h.description, h.phi_q_before, h.phi_q_after);
            }
            continue;
        }

        match parse_intention_block(input) {
            Ok((_, ast)) => {
                println!("  [+] Intenção Compilada: {}", ast.name);

                let phi_before = match sys.sys_check_nucleation() {
                    SyscallResult::WaveCloudStatus(_, phi) => phi,
                    _ => 0.0
                };

                sys_lock.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);

                match sys_lock.sys_tick() {
                    SyscallResult::Success(msg) => {
                        println!("  [OK] {}", msg);
                        let phi_after = match sys_lock.sys_check_nucleation() {
                sys.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);

                match sys.sys_tick() {
                    SyscallResult::Success(msg) => {
                        println!("  [OK] {}", msg);

                        let phi_after = match sys.sys_check_nucleation() {
                            SyscallResult::WaveCloudStatus(_, phi) => phi,
                            _ => 0.0
                        };

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
                        ledger.commit_handover(handover)?;
                    }

                        ledger.commit_handover(handover)?;
                    }
                    SyscallResult::Error(e) => println!("  [ERR] {}", e),
                    _ => ()
                }
            }
            Err(_) => {
                println!("  [ERR] Syntax Error.");
            }
                println!("  [ERR] Falha de sintaxe.");
            }
                println!("  [ERR] Falha de sintaxe. Use: intention <name> {{ target: \"...\" coherence: 0.5 priority: 1 payload: \"...\" }}");
            }
        if input.is_empty() {
            continue;
        }
        if input.is_empty() { continue; }

        let parts: Vec<&str> = input.split_whitespace().collect();
        match parts[0] {
            "create" => {
                if parts.len() < 5 {
                    println!("Uso: create <nome> <coerência> <duração> <prioridade>");
                    continue;
                }
                if parts.len() < 5 { continue; }
                let name = parts[1];
                let coherence: f64 = parts[2].parse().unwrap_or(0.5);
                let duration: u64 = parts[3].parse().unwrap_or(10);
                let priority: i32 = parts[4].parse().unwrap_or(1);
                match sys.sys_create_task(name, coherence, duration, priority) {
                    kernel::syscall::SyscallResult::TaskId(id) => {
                        println!("Tarefa criada com ID {}", id);
                    }
                    SyscallResult::TaskId(id) => println!("Tarefa {} criada", id),
                    _ => println!("Erro ao criar tarefa"),
                }
            }
            "tick" => {
                match sys.sys_tick() {
                    kernel::syscall::SyscallResult::Success(msg) => println!("{}", msg),
                    kernel::syscall::SyscallResult::Error(msg) => println!("Erro: {}", msg),
                    SyscallResult::Success(msg) => println!("{}", msg),
                    SyscallResult::Error(msg) => println!("Erro: {}", msg),
                    _ => (),
                }
            }
            "status" => {
                match sys.sys_coherence_status() {
                    kernel::syscall::SyscallResult::CoherenceUpdate(avail) => {
                        // Para obter φ_q, usamos sys_check_nucleation
                        let nucl = sys.sys_check_nucleation();
                        let (nucleated, phi_q) = match nucl {
                            kernel::syscall::SyscallResult::WaveCloudStatus(n, p) => (n, p),
                            _ => (false, 0.0),
                        };
                        println!("Coerência disponível: {:.3}", avail);
                        println!("φ_q actual: {:.3}", phi_q);
                        if nucleated {
                            println!("⚠️  WAVE-CLOUD NUCLEATION DETECTED");
                        }
                    }
                    _ => (),
                }
            }
            "nucleation" => {
                match sys.sys_check_nucleation() {
                    kernel::syscall::SyscallResult::WaveCloudStatus(nucleated, phi_q) => {
                        println!("φ_q = {:.3} – {}", phi_q, if nucleated { "NUCLEADO" } else { "abaixo do limiar" });
                    }
                    _ => (),
                }
            }
            "handover" => {
                if parts.len() < 3 {
                    println!("Uso: handover <época> <payload>");
                    continue;
                }
                let epoch: u32 = parts[1].parse().unwrap_or(2009);
                let payload = parts[2..].join(" ");
                match sys.sys_handover(epoch, &payload) {
                    kernel::syscall::SyscallResult::Success(msg) => println!("{}", msg),
                    _ => println!("Erro no handover"),
                }
            }
            "exit" | "quit" => break,
                    SyscallResult::CoherenceUpdate(avail) => {
                        match sys.sys_check_nucleation() {
                            SyscallResult::WaveCloudStatus(nucleated, phi_q) => {
                                println!("Coerência disponível: {:.3}", avail);
                                println!("φ_q actual: {:.3}", phi_q);
                                if nucleated { println!("⚠️  WAVE-CLOUD NUCLEATION DETECTED"); }
                            }
                            _ => (),
                        }
                    }
                    _ => (),
                }
            }
            "exit" => break,
            _ => println!("Comando desconhecido"),
        }
    }

    Ok(())
}
