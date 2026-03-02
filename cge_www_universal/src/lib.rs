mod www;
pub use www::*;

pub async fn run() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    tracing::info!("🚀 Starting CGE Alpha - World Wide Web Universal Layer...");

    let _core = www::WWWUniversalCore::bootstrap(None).await?;

    tracing::info!("✅ WWW Universal Core is running");

    tokio::signal::ctrl_c().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::www::WWWConfig;

    #[tokio::test]
    async fn test_core_bootstrap() {
        let mut config = WWWConfig::default();
        config.http_ports = vec![8080, 8081];
        config.websocket_ports = vec![3000, 3001];
        config.quic_ports = vec![4433, 4443];
        let core = www::WWWUniversalCore::bootstrap(Some(config)).await;
        if let Err(e) = &core {
            println!("Bootstrap error: {:?}", e);
        }
        assert!(core.is_ok());
    }
}
