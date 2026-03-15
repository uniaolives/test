use orbvm::network::ssh::SSHServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server = SSHServer::new(3141);
    server.run().await?;
    Ok(())
}
