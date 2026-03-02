use arkhe_quantum::manifold::GlobalManifold;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_nanos()
        .init();

    log::info!("🜁 ARKHE(n) QUANTUM OS – PROTOCOLO Ω+210");

    let manifold = Arc::new(Mutex::new(GlobalManifold::new("localhost").await?));

    let _manifold_server = manifold.clone();
    tokio::spawn(async move {
        log::info!("Servidor QHTTP (Simulado) iniciado");
    });

    arkhe_quantum::asi_core::singularity_engine_loop(manifold).await;
}
