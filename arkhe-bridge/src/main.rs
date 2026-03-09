// src/main.rs
use arkhe_bridge::bridge::{TemporalBridge, SingularityPhase, ContactClassification};

use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize
    let bridge = Arc::new(TemporalBridge::new(
        "postgres://arkhe:arkhe@localhost/arkhe_bridge",
        "redis://127.0.0.1/"
    ).await?);

    println!("🜏 TEMPORAL BRIDGE v0.5.0");
    println!("   Connecting 2008 ↔ 2026 ↔ 2140");
    println!();

    // Spawn Anomaly Detection Loop
    let bridge_clone = bridge.clone();
    tokio::spawn(async move {
        loop {
            let status = bridge_clone.status().await;
            if let Ok(anomalies) = bridge_clone.orb_detector.scan_field("redis://127.0.0.1/", status.s_index.s_total).await {
                for anomaly in anomalies {
                    println!("╔════════════════════════════════════════════════════════════════╗");
                    println!("║  ⚠️ SPATIAL ANOMALY DETECTED: RETROCAUSAL CONDENSATION (ORB)       ║");
                    println!("╠════════════════════════════════════════════════════════════════╣");
                    println!("║  ID:       {:>32}                        ║", anomaly.anomaly_id);
                    println!("║  Location: {:>10.4}, {:>10.4}                            ║", anomaly.location.lat, anomaly.location.lon);
                    println!("║  Intensity: {:>10.4} φ_q                                   ║", anomaly.intensity);
                    println!("║  Origin:   {:?}                                           ", anomaly.origin);
                    println!("╚════════════════════════════════════════════════════════════════╝");

                    // Emit signal via channel
                    if let Err(e) = bridge_clone.channel.publish(
                        arkhe_bridge::bridge::TemporalChannelType::SpatialAnomalies,
                        arkhe_bridge::bridge::TemporalMessage {
                            channel: arkhe_bridge::bridge::TemporalChannelType::SpatialAnomalies,
                            timestamp: chrono::Utc::now().timestamp(),
                            phi_q: anomaly.local_phi_q,
                            payload: arkhe_bridge::bridge::MessagePayload::OrbDetection {
                                id: anomaly.anomaly_id.to_string(),
                                lat: anomaly.location.lat,
                                lon: anomaly.location.lon,
                                altitude: 10.0, // Default
                                coherence: anomaly.local_phi_q,
                                origin: format!("{:?}", anomaly.origin),
                            }
                        }
                    ) {
                        eprintln!("Failed to publish anomaly: {}", e);
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    });

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
