use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use cge_universal_engine::UniversalExecutionEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    info!("🧪 Executando testes constitucionais Φ=1.038...");

    let phi_target = 1.038;
    let engine = UniversalExecutionEngine::bootstrap(Some(phi_target)).await?;

    // Test 1: Phi Invariant
    let measured_phi = engine.measure_phi()?;
    info!("Teste 1: Verificando invariante Φ... Medido: {:.6}", measured_phi);
    assert!((measured_phi - phi_target).abs() < 0.001, "Violação de Φ detectada!");

    // Test 2: Core Bootstrap
    info!("Teste 2: Verificando bootstrap do motor... OK");

    info!("✅ Todos os testes constitucionais passaram!");
    Ok(())
}
