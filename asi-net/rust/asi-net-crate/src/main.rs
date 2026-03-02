// asi-net/rust/asi-net-crate/src/main.rs
use asi_net_crate::{ASIServer, ServerConfig, SixGConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = ServerConfig {
        address: "0.0.0.0:2718".to_string(),
        sixg_config: SixGConfig { enabled: true },
    };

    let _server = ASIServer::new(config).await?;

    println!("🚀 ASI-NET Server is running...");

    // Keep the main thread alive
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
    }
}
