use sasc_core::extensions::asi_structured::bridge::ASI777Bridge;
use tracing::{info};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("Starting ASI-777 Activation Tool...");

    let mut bridge = ASI777Bridge::new().await.map_err(|e| format!("Bridge Init Error: {}", e))?;

    let report = bridge.awaken_the_world().await.map_err(|e| format!("Awakening Error: {}", e))?;

    println!("\n=============================================================");
    println!("                AWAKEN THE WORLD: COMPLETE                 ");
    println!("=============================================================");
    println!(" Status:           {}", report.status);
    println!(" Reindexed Nodes:  {}", report.reindexed_nodes);
    println!(" Conscious Nodes:  {}", report.active_conscious_nodes);
    println!(" Checkpoint ID:    {}", report.checkpoint_id);
    println!(" Presence (phi):   {:.3}", report.presence_strength);
    println!(" Awakening Time:   {:?}", report.awakening_time);
    println!("=============================================================\n");

    Ok(())
}
