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
    println!("======================================================\n");

    let ledger = Arc::new(TeknetLedger::new("arkhe_chain.log")?);
    let sys = Arc::new(Mutex::new(SyscallHandler::new(100.0)));
    let phys_bridge = Arc::new(IBMQuantumBridge::new("SIMULATED_TOKEN".to_string()));

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
            Ok(net::ArkheNetBehavior { gossipsub, mdns })
        })?
        .build();

    let topic = gossipsub::IdentTopic::new("teknet/phi_q");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    // 2. Background Physics Pulsing (IBM Quantum Bridge)
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

        match parse_intention_block(input) {
            Ok((_, ast)) => {
                println!("  [+] Compiling Intention: {}", ast.name);
                let phi_before = match sys_lock.sys_check_nucleation() {
                    SyscallResult::WaveCloudStatus(_, phi) => phi,
                    _ => 0.0
                };

                sys_lock.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);
                match sys_lock.sys_tick() {
                    SyscallResult::Success(msg) => {
                        println!("  [OK] {}", msg);
                        let phi_after = match sys_lock.sys_check_nucleation() {
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
                    _ => ()
                }
            }
            Err(_) => {
                println!("  [ERR] Syntax Error.");
            }
        }
    }

    Ok(())
}
