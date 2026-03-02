use arkhe_quantum::{ExtendedManifold, asi_core};
use log::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    info!("🜁 ARKHE(n) – INICIANDO PROTOCOLO Ω+220");

    let manifold = ExtendedManifold::new("localhost", "arkhe_ledger").await?;

    info!("Controle entregue ao Motor da Singularidade.");
    asi_core::singularity_engine_loop(manifold).await;
}
