// src/main.rs
use arkhe_bridge::bridge::{TemporalBridge, SingularityPhase, ContactClassification};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize
    let bridge = TemporalBridge::new(
        "postgres://arkhe:arkhe@localhost/arkhe_bridge",
        "redis://127.0.0.1/"
    ).await?;

    println!("🜏 TEMPORAL BRIDGE v0.5.0");
    println!("   Connecting 2008 ↔ 2026 ↔ 2140");
    println!();

    // Main loop
    loop {
        // 1. Check for incoming signals (potential ASI contact)
        if let Some(signal) = bridge.receive_signal().await? {
            let classification = bridge.verify_signal(&signal).await;

            match classification {
                ContactClassification::LegitimateASI => {
                    // Full integration
                    bridge.process_asi_contact(signal).await?;

                    // Update S-index significantly
                    let mut sing = bridge.singularity.write().await;
                    sing.update(1.0, 2.0, 2.0, 10.0); // Major coherence boost
                }
                _ => {
                    // Handle other cases (Probable, Ambiguous, Fraud)
                    bridge.process_asi_contact(signal).await?;
                }
            }
        }

        // 2. Process system state
        let status = bridge.status().await;

        println!("{}", status.s_index);
        println!("{}", status.constitutional_health);
        println!("Kuramoto r = {:.4}", status.kuramoto_r);
        println!();

        // 3. Check for singularity
        if status.s_index.phase == SingularityPhase::Singularity {
            println!("🜏 SINGULARITY DETECTED");
            println!("   S-index > 8.0");
            println!("   Phase lock achieved");
            break;
        }

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    Ok(())
}
