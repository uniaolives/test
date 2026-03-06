mod net;
mod phys;
mod ledger;
mod physics;
mod kernel;
mod constitution;
mod ffi;
mod stats;

use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::ledger::Ledger;
use crate::net::node::P2PNode;
use crate::phys::ibm_client::QuantumAntenna;

pub type ArkheError = String; // Simplificado para o binário

#[tokio::main]
async fn main() {
    println!("🜁 ArkheOS Persistent Kernel v0.5 (Multi-Dimensional)");

    let ledger = Arc::new(Mutex::new(Ledger::create("./data/teknet.bin").unwrap()));

    let node = P2PNode::new(7000, ledger.clone());
    tokio::spawn(async move {
        node.run_server().await;
    });

    let antenna = QuantumAntenna::new("SIMULATED_TOKEN".to_string());

    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(30)).await;
            if let Ok(real_phi_q) = antenna.measure_vacuum_quality("ibm_brisbane").await {
                println!("[SYSTEM] Physical vacuum recalibrated: φ_q = {:.3}", real_phi_q);
            }
        }
    });

    loop {
        sleep(Duration::from_secs(1)).await;
    }
}
