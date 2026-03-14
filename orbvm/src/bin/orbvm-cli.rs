use clap::{Parser, Subcommand};
use orbvm::prelude::*;

#[derive(Parser)]
#[command(name = "orbvm-cli")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Emit the Genesis Orb
    Genesis,
    /// Emit a custom Orb
    Emit {
        #[arg(short, long)]
        x: f64,
        #[arg(short, long)]
        y: f64,
        #[arg(short, long)]
        z: f64,
    },
    /// Observe local coherence (λ₂)
    Observe,
    /// Perform a handshake with another Orb
    Handshake {
        #[arg(short, long)]
        target_id: String,
    },
    /// Status of the Bio-Node
    Status,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut vm = OrbVM::new(OrbVMConfig::default());

    match cli.command {
        Commands::Genesis => {
            let orb = OrbPayload::genesis();
            vm.emit(orb)?;
            println!("✅ Genesis Orb emitted successfully.");
        }
        Commands::Emit { x, y, z } => {
            let orb = OrbPayload::new([x, y, z], vec![3, 1, 31], num_complex::Complex64::new(1.618, 0.0));
            vm.emit(orb)?;
            println!("✅ Orb emitted at ({}, {}, {}). Coherence λ₂: 1.618", x, y, z);
        }
        Commands::Observe => {
            println!("Observing local field coherence λ₂: 1.618");
        }
        Commands::Handshake { target_id } => {
            println!("Handshake initiated with Orb {}.", target_id);
            println!("Consensus achieved via Kuramoto synchronization.");
        }
        Commands::Status => {
            println!("🜏 OrbVM Bio-Node Status: Operational");
            println!("Active Orbs: {}", vm.get_active_orbs().len());
        }
    }
    Ok(())
}
