use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    request_id: String,
    #[arg(long)]
    nexus_phi: f64,
    #[arg(long)]
    decision: String,
    #[arg(long)]
    constraints: String,
    #[arg(long)]
    effective_immediately: bool,
}

fn main() {
    let args = Args::parse();
    println!("Architect Ruling for request: {}", args.request_id);
    println!("Nexus Φ: {}", args.nexus_phi);
    println!("Decision: {}", args.decision);
    println!("Constraints: {}", args.constraints);
    println!("Effective immediately: {}", args.effective_immediately);
}
