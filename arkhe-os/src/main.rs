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

    // 3. Iniciar Antena Biocibernética
    let bio = BioAntenna::new(7001);
    let kernel_clone = kernel.clone();
    tokio::spawn(async move {
        bio.run(kernel_clone).await;
    });

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

#[cfg(test)]
mod tests;

use kernel::syscall::{SyscallHandler, SyscallResult};
use intention::parser::parse_intention_block;
use arkhe_db::ledger::TeknetLedger;
use arkhe_db::schema::{Handover, HandoverStatus};
use phys::ibm_sensor::IBMQuantumBridge;
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
    println!(" 🜁 ArkheOS Node v0.4 — Teknet Integrated Stack");
    println!(" [L] LITE integrated with [D] DATA");
    println!(" Digite sua intenção em intention-lang ou 'exit' para sair.");
    println!("======================================================\n");

    let ledger = Arc::new(TeknetLedger::new("arkhe_chain.log")?);
    let sys = Arc::new(Mutex::new(SyscallHandler::new(100.0)));
    let phys_bridge = Arc::new(IBMQuantumBridge::new("SIMULATED_TOKEN".to_string()));

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

    let topic = gossipsub::IdentTopic::new("teknet/phi_q");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

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
