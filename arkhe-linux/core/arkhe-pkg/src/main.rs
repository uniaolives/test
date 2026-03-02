use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(name = "arkhe-pkg")]
#[command(about = "Arkhe(n) Package Manager (Thermodynamic)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Instala um pacote
    Install { package_name: String },
    /// Remove um pacote
    Remove { package_name: String },
    /// Lista pacotes instalados
    List,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Package {
    name: String,
    version: String,
    entropy: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Manifest {
    packages: Vec<Package>,
}

const MANIFEST_PATH: &str = "/var/lib/arkhe/packages.json";

fn load_manifest() -> Manifest {
    if Path::new(MANIFEST_PATH).exists() {
        let content = fs::read_to_string(MANIFEST_PATH).unwrap_or_else(|_| "{\"packages\": []}".to_string());
        serde_json::from_str(&content).unwrap_or(Manifest { packages: Vec::new() })
    } else {
        Manifest { packages: Vec::new() }
    }
}

fn save_manifest(manifest: &Manifest) -> anyhow::Result<()> {
    let parent = Path::new(MANIFEST_PATH).parent().expect("Invalid manifest path");
    fs::create_dir_all(parent)?;
    let content = serde_json::to_string_pretty(manifest)?;
    fs::write(MANIFEST_PATH, content)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut manifest = load_manifest();

    match cli.command {
        Commands::Install { package_name } => {
            if manifest.packages.iter().any(|p| p.name == package_name) {
                println!("Pacote {} já está instalado.", package_name);
            } else {
                println!("Instalando {}...", package_name);
                manifest.packages.push(Package {
                    name: package_name.clone(),
                    version: "1.0.0".to_string(),
                    entropy: 0.01,
                });
                save_manifest(&manifest)?;
                println!("✅ Pacote {} instalado com sucesso.", package_name);
            }
        }
        Commands::Remove { package_name } => {
            let initial_len = manifest.packages.len();
            manifest.packages.retain(|p| p.name != package_name);
            if manifest.packages.len() < initial_len {
                save_manifest(&manifest)?;
                println!("✅ Pacote {} removido.", package_name);
            } else {
                println!("Pacote {} não encontrado.", package_name);
            }
        }
        Commands::List => {
            println!("Pacotes instalados:");
            for p in &manifest.packages {
                println!("  - {} (v{}) [Entropia: {:.4}]", p.name, p.version, p.entropy);
            }
            if manifest.packages.is_empty() {
                println!("  (nenhum pacote instalado)");
            }
        }
    }

    Ok(())
}
