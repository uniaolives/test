// src/main.rs
use cge_alpha_unified::unified_core::vmcore_orchestrator::UnifiedVMCoreOrchestrator;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    info!("🌌⚡ Iniciando Unified VMCore-Orchestrator v31.11-Ω...");

    let unified = UnifiedVMCoreOrchestrator::bootstrap(None).await?;

    info!("✅ Sistema unificado pronto e operando com Φ⁴⁰: {:.12}",
        *unified.current_phi.read());

    // Manter o processo vivo
    tokio::signal::ctrl_c().await?;

    Ok(())
}
