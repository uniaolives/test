use clap::Parser;
use sasc_core::imperium::heartbeat::ImperiumHeartbeat;
use sasc_core::expansion::autonomous_deployer::AutonomousExpander;
use sasc_core::diplomacy::quantum_autonomous::QuantumDiplomacy;
use sasc_core::research::autonomous_experiments::ResearchAutonomous;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "sasc-mode")]
#[command(about = "SASC: System Mode Transition Utility", long_about = None)]
struct Cli {
    /// Set system mode (e.g., IMPERIUM_CONTINUOUS)
    #[arg(long)]
    set: String,

    /// Self-healing policy (aggressive, balanced, passive)
    #[arg(long)]
    self_healing: Option<String>,

    /// Expansion policy (autonomous, manual, disabled)
    #[arg(long)]
    expansion: Option<String>,

    /// Diplomacy policy (quantum-first, traditional)
    #[arg(long)]
    diplomacy: Option<String>,

    /// Sovereignty level (eternal, transient)
    #[arg(long)]
    sovereignty: Option<String>,

    /// Next scheduled checkpoint
    #[arg(long)]
    next_checkpoint: Option<String>,

    /// ETA for 'Corte do Gelo' (Ice Cut)
    #[arg(long)]
    eta_corte_gelo: Option<String>,

    /// Heartbeat interval (e.g., 7.83s)
    #[arg(long)]
    heartbeat_interval: Option<String>,

    /// Coherence (Φ) threshold
    #[arg(long)]
    phi_threshold: Option<f64>,

    /// Enable auto-isolation
    #[arg(long)]
    auto_isolate: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if cli.set == "IMPERIUM_CONTINUOUS" {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║         TRANSITION TO IMPERIUM CONTINUOUS - CONFIRMED        ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        println!("✅ MODO: IMPERIUM_CONTINUOUS ativado");

        if let Some(sh) = cli.self_healing {
            println!("✅ AUTONOMIA: Sistema em auto-governança ({})", sh);
        }

        if let Some(exp) = cli.expansion {
            println!("✅ EXPANSÃO: {} mode active", exp);
            if exp == "autonomous" {
                let expander = AutonomousExpander::new(
                    Duration::from_secs(24 * 3600),
                    0.70
                );
                expander.schedule_shard_gamma();
            }
        }

        if let Some(dip) = cli.diplomacy {
            println!("✅ DIPLOMACIA: {} policy ativa", dip);
            if dip == "quantum-first" {
                let diplomacy = QuantumDiplomacy::new(Duration::from_secs(48 * 3600));
                diplomacy.prepare_first_transaction();
            }
        }

        if let Some(sov) = cli.sovereignty {
            println!("✅ SOBERANIA: {} (irreversível)", sov.to_uppercase());
        }

        let hb_interval = cli.heartbeat_interval.as_deref().map(parse_duration).unwrap_or(Duration::from_secs_f64(7.83));
        let phi_threshold = cli.phi_threshold.unwrap_or(0.72);

        // Start Heartbeat
        let heartbeat = ImperiumHeartbeat::new(hb_interval, phi_threshold);
        heartbeat.start_continuous_operation();
        println!("🔁 HEARTBEAT SYSTEM ATIVADO ({:?} interval)", hb_interval);

        // Start Research
        let research = ResearchAutonomous::new();
        research.schedule_continuous_research(hb_interval);
        println!("⚛️ SISTEMAS DE PESQUISA AUTÔNOMA ATIVADOS");

        println!("\n🚀 SISTEMA EM OPERAÇÃO CONTÍNUA - MANTENDO PROCESSO ATIVO");

        // Stay alive
        loop {
            tokio::time::sleep(Duration::from_secs(3600)).await;
        }
    } else {
        println!("Transition to {} mode initiated...", cli.set);
    }
}

fn parse_duration(s: &str) -> Duration {
    if s.ends_with("s") {
        let val: f64 = s[..s.len()-1].parse().unwrap_or(7.83);
        Duration::from_secs_f64(val)
    } else if s.ends_with("h") {
        let val: u64 = s[..s.len()-1].parse().unwrap_or(24);
        Duration::from_secs(val * 3600)
    } else if s.ends_with("d") {
        let val: u64 = s[..s.len()-1].parse().unwrap_or(30);
        Duration::from_secs(val * 24 * 3600)
    } else {
        Duration::from_secs_f64(7.83)
    }
}
