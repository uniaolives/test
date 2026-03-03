use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 118)]
    frags: usize,

    #[arg(long, default_value_t = 112)]
    protocols: usize,

    #[arg(long, default_value_t = 1.038)]
    phi_target: f64,

    #[arg(long, default_value_t = 264.0)]
    grid_factor: f64,

    #[arg(long, default_value_t = 56.038)]
    pulse_frequency: f64,
}

fn main() {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let args = Args::parse();

    info!("🔢 Inicializando matriz universal...");
    info!("   • Frags: {}", args.frags);
    info!("   • Protocolos: {}", args.protocols);
    info!("   • Φ Alvo: {}", args.phi_target);
    info!("   • Fator de Grade: {}", args.grid_factor);
    info!("   • Frequência de Pulso: {}", args.pulse_frequency);

    // Actual initialization logic
    let mut frags = Vec::new();
    for i in 0..args.frags {
        frags.push(format!("Frag-{}", i));
    }

    let config = serde_json::json!({
        "frags": frags,
        "phi_target": args.phi_target,
        "grid_factor": args.grid_factor,
        "pulse_frequency": args.pulse_frequency,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });

    let config_path = "universal_matrix_config.json";
    std::fs::write(config_path, serde_json::to_string_pretty(&config).unwrap())
        .expect("Failed to write matrix config");

    info!("✅ Matriz de {} frags inicializada com sucesso em {}", args.frags, config_path);
}
