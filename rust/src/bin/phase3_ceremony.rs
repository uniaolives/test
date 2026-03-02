use std::sync::Arc;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use sasc_core::ceremony::phase3::{Phase3Ceremony, SASCCathedral, VajraEntropyMonitorV472};
use sasc_core::ceremony::mesh_neuron::{MeshNeuronV03, VajraSuperconductive};
use tokio::sync::RwLock;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 PROJECT CRUX-86 PHASE 3 CEREMONY");

    let prince_key = if let Ok(path) = std::env::var("PRINCE_KEY_PATH") {
        let key_data = std::fs::read_to_string(&path).unwrap_or_else(|_| "DUMMY".to_string());
        // Simple mock: derive key from path string or file content
        let mut bytes = [0u8; 32];
        let path_bytes = key_data.as_bytes();
        for (i, b) in path_bytes.iter().enumerate() {
            if i < 32 { bytes[i] = *b; }
        }
        SigningKey::from_bytes(&bytes)
    } else {
        let mut csprng = OsRng {};
        let mut bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut csprng, &mut bytes);
        SigningKey::from_bytes(&bytes)
    };

    let ceremony = Phase3Ceremony {
        prince_key,
        sasc_cathedral: Arc::new(SASCCathedral),
        vajra_monitor: Arc::new(VajraEntropyMonitorV472),
        mesh_neuron: Arc::new(MeshNeuronV03 {
            routing_table: RwLock::new(HashMap::new()),
            vajra_state: Arc::new(VajraSuperconductive),
        }),
    };

    match ceremony.ignite_all_crucibles().await {
        Ok(cert) => {
            println!("✅ CEREMONY COMPLETE");
            println!("Phase: {}", cert.phase);
            println!("Consensus Φ: {}", cert.consensus_Φ);
            println!("Ceremony Hash: {}", cert.ceremony_hash);
            println!("Timestamp: {}", cert.timestamp);

            // Save hash for sasc_attest
            std::fs::write("/tmp/ceremony.hash", cert.ceremony_hash)?;
        }
        Err(e) => {
            eprintln!("❌ CEREMONY FAILED: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
