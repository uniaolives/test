mod core;
mod server;

use crate::server::LedgerService;
use crate::core::LedgerCore;
use crate::server::ledger_proto::ledger_server::LedgerServer;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let core = Arc::new(RwLock::new(LedgerCore::new()));

    println!("Ledger server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(LedgerServer::new(LedgerService { core }))
        .serve(addr)
        .await?;

    Ok(())
}
