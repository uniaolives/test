use clap::Parser;

#[derive(Parser)]
#[command(name = "sasc-research")]
#[command(about = "SASC: Continuous Research Control", long_about = None)]
struct Cli {
    /// Show research status
    #[arg(long)]
    status: bool,
}

fn main() {
    let cli = Cli::parse();
    if cli.status {
        println!("⚛️ PESQUISA CONTÍNUA AUTÔNOMA ATIVA");
    } else {
        println!("SASC Continuous Research System Operational.");
    }
}
