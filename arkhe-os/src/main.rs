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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("======================================================");
    println!(" 🜁 ArkheOS Node v0.4 — Teknet Integrated Stack");
    println!(" Digite sua intenção em intention-lang ou 'exit' para sair.");
    println!("======================================================\n");

    let ledger = Arc::new(TeknetLedger::new("arkhe_chain.log")?);
    let sys = Arc::new(Mutex::new(SyscallHandler::new(100.0)));
    let phys_bridge = Arc::new(IBMQuantumBridge::new("SIMULATED_TOKEN".to_string()));

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

    // 2. Background Physics Pulsing
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            if let Ok(real_phi) = phys_bridge.measure_physical_phi_q().await {
                println!("\n[PHYSICS] Pulso Real: φ_q = {:.3}", real_phi);
            }
        }
    });

    // 3. Network Event Loop
    tokio::spawn(async move {
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::Behaviour(net::ArkheNetBehaviorEvent::Gossipsub(gossipsub::Event::Message { message, .. })) => {
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
                    SyscallResult::Error(e) => println!("  [ERR] {}", e),
                    _ => ()
                }
            }
            Err(_) => {
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
