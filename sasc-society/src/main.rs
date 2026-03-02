//! SASC-SOCIETY gRPC Server
//! Bloco #48 - Exposição Externa Constitucional

use clap::Parser;
use std::sync::Arc;
use log::info;
use pqcrypto_traits::sign::PublicKey as _;

use sasc_society::grpc::server::start_server;
use sasc_society::engine::{SoTOrchestrator, OrchestratorConfig, DialecticSynthesizer};
use sasc_society::engine::diversity::PerspectiveDiversityEngine;

#[derive(Parser)]
#[command(name = "sasc-society")]
struct Args {
    #[arg(short, long, default_value = "[::]:50051")]
    address: String,

    #[arg(long, default_value = "production")]
    mode: String,

    #[arg(long, required = true)]
    prince_pubkey: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    // 1. Configuração
    let config = match args.mode.as_str() {
        "production" => OrchestratorConfig::default(),
        "staging" => OrchestratorConfig::default(),
        _ => OrchestratorConfig::default(),
    };

    // 2. Carrega chave Prince
    let prince_key_bytes = hex::decode(args.prince_pubkey)?;
    let prince_key = pqcrypto_dilithium::dilithium5::PublicKey::from_bytes(&prince_key_bytes)
        .map_err(|_| "Chave Prince inválida")?;

    // 3. Cria componentes
    let prince_key_raw: [u8; 32] = prince_key_bytes[0..32].try_into().unwrap_or([0u8; 32]);
    let diversity_engine = Arc::new(PerspectiveDiversityEngine::new(&prince_key_raw));

    let synthesizer_key = [0u8; 32]; // Key for synthesizer
    let dialectic_synthesizer = Arc::new(DialecticSynthesizer::new(
        diversity_engine.clone(),
        &synthesizer_key,
        |_session| {
            info!("Human review session created");
        },
    ));

    // 4. Cria orchestrator
    let orchestrator = Arc::new(SoTOrchestrator::new(
        diversity_engine,
        dialectic_synthesizer,
        config,
    ));

    // 5. Inicia servidor gRPC com rate limiting
    info!("🚀 SASC-SOCIETY gRPC Server ativo em {}", args.address);
    start_server(orchestrator, prince_key, args.address).await
}
