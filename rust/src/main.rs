// rust/src/main.rs
use sasc_core::error::{ResilientError, ResilientResult};
use sasc_core::wallet::WalletManager;
use sasc_core::checkpoint::{CheckpointManager, CheckpointTrigger};
use sasc_core::network::{ArweaveClient, NostrClient};
use sasc_core::runtime::backend::{AnthropicBackend, BackendConfig, RuntimeBackend};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "Resilient Agent")]
#[command(version = "1.0")]
#[command(about = "AI agent with persistent state via Arweave", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, default_value = ".config/resilient-agent")]
    #[allow(dead_code)]
    config_dir: PathBuf,

    #[arg(long, default_value = "info")]
    #[allow(dead_code)]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new resilient agent
    Init {
        #[arg(long)]
        agent_id: String,

        #[arg(long)]
        #[allow(dead_code)]
        wallet_path: Option<PathBuf>,
    },

    /// Run the agent interactively
    Run {
        #[arg(long)]
        agent_id: String,

        #[arg(long, default_value = "turbo")]
        network: String,

        #[arg(long)]
        checkpoint_every: Option<u64>, // Mensagens entre checkpoints
    },

    /// Create a checkpoint of current state
    Checkpoint {
        #[arg(long)]
        #[allow(dead_code)]
        tx_id: Option<String>, // Se fornecido, restaura primeiro
    },

    /// Restore agent state from a transaction
    Restore {
        #[allow(dead_code)]
        tx_id: String,
    },

    /// Show agent status and info
    Status,

    /// Export agent state for backup
    Export {
        #[allow(dead_code)]
        output: PathBuf,
    },
}

struct ResilientAgentApp {
    wallet: WalletManager,
    checkpoint_manager: Option<CheckpointManager>,
    #[allow(dead_code)]
    runtime: Option<AnthropicBackend>,
    agent_id: Option<String>,
}

impl ResilientAgentApp {
    async fn new() -> ResilientResult<Self> {
        let wallet = WalletManager::new(None).await?;

        Ok(Self {
            wallet,
            checkpoint_manager: None,
            runtime: None,
            agent_id: None,
        })
    }

    async fn initialize_agent(&mut self, agent_id: &str, _wallet_path: Option<PathBuf>) -> ResilientResult<()> {
        info!("Initializing resilient agent: {}", agent_id);

        // 1. Configurar rede
        let arweave_client = ArweaveClient::new();
        let nostr_client = NostrClient::new(vec![]).await?;

        // 2. Criar gerenciador de checkpoint
        let checkpoint_manager = CheckpointManager::new(
            agent_id,
            self.wallet.clone(),
            arweave_client,
            nostr_client,
        ).await?;

        // 3. Configurar runtime (backend de IA)
        let mut runtime = AnthropicBackend::new();

        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .unwrap_or_else(|_| "mock-key".to_string());

        let backend_config = BackendConfig {
            api_key: Some(api_key),
            model: "claude-3-5-sonnet-20241022".to_string(),
            endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            max_tokens: 4096,
            temperature: 0.7,
        };

        runtime.initialize(backend_config)?;

        // 4. Atualizar estado da aplicação
        self.checkpoint_manager = Some(checkpoint_manager);
        self.runtime = Some(runtime);
        self.agent_id = Some(agent_id.to_string());

        info!("Agent {} initialized successfully", agent_id);
        Ok(())
    }

    async fn run_interactive(&mut self, checkpoint_every: Option<u64>) -> ResilientResult<()> {
        let agent_id = self.agent_id.as_ref()
            .ok_or_else(|| ResilientError::Configuration("Agent not initialized".to_string()))?;

        info!("Starting interactive session for agent: {}", agent_id);

        let _message_count = 0;
        let _checkpoint_interval = checkpoint_every.unwrap_or(10);

        println!("Resilient Agent Interactive Shell. Type '/exit' to quit.");
        // Mocking interaction loop for non-interactive environment
        println!("> Hello!");
        println!("Mock response from Anthropic");

        // Create final checkpoint
        self.create_checkpoint(CheckpointTrigger::Manual).await?;

        Ok(())
    }

    async fn create_checkpoint(&mut self, trigger: CheckpointTrigger) -> ResilientResult<()> {
        let checkpoint_manager = self.checkpoint_manager.as_mut()
            .ok_or_else(|| ResilientError::Configuration("Checkpoint manager not initialized".to_string()))?;

        let result = checkpoint_manager.checkpoint(trigger).await?;

        info!("Checkpoint created successfully");
        info!("  TX ID: {}", result.tx_id);
        info!("  Size: {} bytes", result.size_bytes);
        info!("  Cost: {} winston", result.cost_winston);
        info!("  Strategy: {:?}", result.strategy_used);

        Ok(())
    }

    async fn show_status(&self) -> ResilientResult<()> {
        let agent_id = self.agent_id.as_ref()
            .ok_or_else(|| ResilientError::Configuration("Agent not initialized".to_string()))?;

        let address = self.wallet.get_address().await?;
        let balance = self.wallet.sync_balance().await?;

        println!("Resilient Agent Status");
        println!("======================");
        println!("Agent ID: {}", agent_id);
        println!("Wallet: {}", address);
        println!("Balance: {} winston", balance);
        println!("Network: {:?}", self.wallet.network());

        Ok(())
    }
}

#[tokio::main]
async fn main() -> ResilientResult<()> {
    // Configurar logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    let cli = Cli::parse();

    let mut app = ResilientAgentApp::new().await?;

    match cli.command {
        Commands::Init { agent_id, wallet_path } => {
            app.initialize_agent(&agent_id, wallet_path).await?;
            app.create_checkpoint(CheckpointTrigger::Manual).await?;
        }

        Commands::Run { agent_id, network, checkpoint_every } => {
            std::env::set_var("ARWEAVE_NETWORK", network);
            app.initialize_agent(&agent_id, None).await?;
            app.run_interactive(checkpoint_every).await?;
        }

        Commands::Checkpoint { .. } => {
            app.initialize_agent("default-agent", None).await?;
            app.create_checkpoint(CheckpointTrigger::Manual).await?;
        }

        Commands::Restore { .. } => {
            println!("Restore command not implemented yet");
        }

        Commands::Status => {
            app.initialize_agent("default-agent", None).await?;
            app.show_status().await?;
        }

        Commands::Export { .. } => {
            println!("Export command not implemented yet");
        }
    }

    Ok(())
}
