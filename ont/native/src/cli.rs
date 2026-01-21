use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "onto")]
#[command(about = "Ontology CLI v0.7.0")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Compile Ontology source code
    Compile {
        /// Input file
        input: String,

        /// Output directory
        #[arg(short, long)]
        output: Option<String>,

        /// Target platform
        #[arg(long, default_value = "solidity")]
        target: String,
    },

    /// Deploy a contract
    Deploy {
        /// Contract bytecode
        #[arg(long)]
        bytecode: String,

        /// RPC URL
        #[arg(long)]
        rpc_url: String,

        /// Private key
        #[arg(long)]
        private_key: String,
    },

    /// Iniciar Auditoria Contínua de Produção
    Audit {
        /// Endereço do contrato a ser auditado
        #[arg(short, long)]
        contract: String,

        /// Seed quântica do Arquiteto-Ω (hex)
        #[arg(short, long)]
        quantum_seed: String,

        /// RPC URL da rede
        #[arg(short, long, default_value = "http://localhost:8545")]
        rpc: String,

        /// Chave privada para reportar falhas (opcional)
        #[arg(short, long)]
        private_key: Option<String>,

        /// Intervalo entre ciclos (segundos)
        #[arg(short, long, default_value = "30")]
        interval: u64,

        /// Modo mobile (requer JNI env vars)
        #[arg(long)]
        mobile: bool,

        /// Modo daemon (executar em background)
        #[arg(long)]
        daemon: bool,
    },

    /// Check current audit status (Mock)
    AuditStatus {
        /// Contract address
        #[arg(long)]
        contract: String,

        /// RPC URL
        #[arg(long)]
        rpc_url: String,
    },
}
