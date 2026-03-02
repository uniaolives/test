use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use cge_universal_engine::integration::www_universal_integration::WWWUniversalIntegration;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 8088)]
    engine_port: u16,

    #[arg(long, default_value_t = 8080)]
    www_port: u16,

    #[arg(long, default_value_t = 1.038)]
    phi_target: f64,

    #[arg(long, default_value = "1s")]
    sync_interval: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let args = Args::parse();

    info!("🌉 Estabelecendo ponte Universal ↔ WWW...");
    info!("   • Engine Port: {}", args.engine_port);
    info!("   • WWW Port: {}", args.www_port);
    info!("   • Φ Alvo: {}", args.phi_target);

    let _integration = WWWUniversalIntegration::create_universal_bridge(args.phi_target).await?;

    info!("✅ Ponte Universal ↔ WWW ativa");

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
    }
}
