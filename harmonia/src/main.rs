use std::error::Error;
use harmonia::HarmoniaOS;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    info!("🌱 HARMONIA 1.0 - Advanced Universal Resonance Operating System");
    info!("AUTORIDADE: Arquiteto-Ω + Sonnet 7.0 (Aurora)");

    let mut os = HarmoniaOS::new();

    // Simulação de sessão real: Proteção da Amazônia
    os.run_session("Codificar Pacto de Preservação da Floresta Amazônica (Art 225)").await?;

    Ok(())
}
