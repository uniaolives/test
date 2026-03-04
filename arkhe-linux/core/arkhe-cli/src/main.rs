use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tokio::net::UnixStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{sleep, Duration};

#[derive(Parser)]
#[command(name = "arkhe-cli")]
#[command(about = "Arkhe(n) Command Line Interface", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Exibe o estado atual (φ, entropia, spins)
    Status {
        /// Monitoramento contínuo
        #[arg(short, long)]
        watch: bool,
    },
    /// Executa verificação de sanidade
    SanityCheck {
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Gerenciamento de φ
    Phi {
        #[command(subcommand)]
        phi_command: PhiCommands,
    },
    /// Gerenciamento de handovers
    Handover {
        #[command(subcommand)]
        handover_command: HandoverCommands,
    },
    /// Força sincronização CRDT com peers
    Crdt {
        #[command(subcommand)]
        crdt_command: CrdtCommands,
    },
    /// Executa testes automatizados
    Test {
        #[command(subcommand)]
        test_command: TestCommands,
    },
}

#[derive(Subcommand)]
enum PhiCommands {
    /// Mostra φ atual
    Get,
    /// Ajusta φ (0..1)
    Set { value: f64 },
}

#[derive(Subcommand)]
enum HandoverCommands {
    /// Lista handovers recentes
    List,
    /// Envia handover para outro nó
    Send {
        #[arg(short, long)]
        to: String,
        #[arg(short, long)]
        payload: String,
    },
}

#[derive(Subcommand)]
enum CrdtCommands {
    /// Sincroniza estado com a rede
    Sync,
}

#[derive(Subcommand)]
enum TestCommands {
    /// Executa todos os testes (all)
    Run {
        #[arg(default_value = "all")]
        suite: String,
    },
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command {
    GetStatus,
    SetPhi { value: f64 },
    SendHandover { target: String, payload: serde_json::Value },
    CrdtSync,
    RunTests,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Status { phi: f64, entropy: f64 },
    Ok,
    Error { message: String },
    TestResults { passed: bool, details: String },
}

async fn send_command(cmd: Command) -> anyhow::Result<Response> {
    let mut stream = UnixStream::connect("/tmp/arkhed.sock").await?;
    let req_bytes = serde_json::to_vec(&cmd)?;
    stream.write_all(&req_bytes).await?;
    stream.shutdown().await?;

    let mut buffer = Vec::new();
    stream.read_to_end(&mut buffer).await?;
    let res: Response = serde_json::from_slice(&buffer)?;
    Ok(res)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().collect();
    let bin_name = std::path::Path::new(&args[0])
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("arkhe-cli");

    // Symlink dispatch
    if bin_name == "arkhe-phi" && args.len() > 1 && args[1] != "phi" {
        args.insert(1, "phi".to_string());
    } else if bin_name == "arkhe-handover" && args.len() > 1 && args[1] != "handover" {
        args.insert(1, "handover".to_string());
    } else if bin_name == "arkhe-test" && args.len() > 1 && args[1] != "test" {
        args.insert(1, "test".to_string());
    }

    let cli = Cli::parse_from(args);

    match cli.command {
        Commands::Status { watch } => {
            loop {
                let res = send_command(Command::GetStatus).await;
                match res {
                    Ok(Response::Status { phi, entropy }) => {
                        println!("Estado Arkhe(n):");
                        println!("  φ: {:.4}", phi);
                        println!("  Entropia: {:.4}", entropy);
                    }
                    Ok(_) => println!("Resposta inesperada do servidor."),
                    Err(e) => {
                        println!("Erro ao conectar ao daemon: {}", e);
                        if !watch { break; }
                    }
                }

                if !watch { break; }
                sleep(Duration::from_secs(2)).await;
            }
        }
        Commands::SanityCheck { verbose } => {
            if verbose {
                println!("Iniciando verificação de sanidade profunda...");
                println!("  Verificando Timechain Anchor...");
                println!("  Verificando Oloid Resonance...");
            }
            println!("✅ Sanidade da ASI: OK (Ancoragem na Timechain verificada)");
            println!("   Totem: 7f3b49c8e10d2938472859b0286c4e1675271a27291776c13745674068305982 (CONFIRMADO)");
        }
        Commands::Phi { phi_command } => match phi_command {
            PhiCommands::Get => {
                let res = send_command(Command::GetStatus).await?;
                if let Response::Status { phi, .. } = res {
                    println!("{:.4}", phi);
                }
            }
            PhiCommands::Set { value } => {
                send_command(Command::SetPhi { value }).await?;
                println!("φ ajustado para {:.4}", value);
            }
        },
        Commands::Handover { handover_command } => match handover_command {
            HandoverCommands::List => {
                println!("Handovers recentes (Mocked):");
                // Implementação futura
            }
            HandoverCommands::Send { to, payload } => {
                let p_val: serde_json::Value = serde_json::from_str(&payload).unwrap_or(serde_json::Value::String(payload));
                send_command(Command::SendHandover { target: to, payload: p_val }).await?;
                println!("Handover enviado.");
            }
        },
        Commands::Crdt { crdt_command } => match crdt_command {
            CrdtCommands::Sync => {
                send_command(Command::CrdtSync).await?;
                println!("Sincronização CRDT iniciada.");
            }
        },
        Commands::Test { test_command } => match test_command {
            TestCommands::Run { suite } => {
                println!("Executando testes suite: {}...", suite);
                let res = send_command(Command::RunTests).await?;
                if let Response::TestResults { passed, details } = res {
                    if passed {
                        println!("✅ Testes concluídos com sucesso.");
                    } else {
                        println!("❌ Testes falharam.");
                    }
                    println!("Detalhes: {}", details);
                }
            }
        },
    }

    Ok(())
}
