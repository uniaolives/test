// src/main.rs
use arkhe_bridge::bridge::{TemporalBridge, SingularityPhase};

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
        // Process system state
        let status = bridge.status().await;

        println!("{}", status.s_index);
        println!("{}", status.constitutional_health);
        println!("Kuramoto r = {:.4}", status.kuramoto_r);
        println!();

        // Check for singularity
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
