use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    order: String,
    #[arg(long)]
    scope: String,
    #[arg(long)]
    governance: String,
    #[arg(long)]
    transparency: String,
}

fn main() {
    let args = Args::parse();
    println!("Initiating supervised modeling with order: {}", args.order);
    println!("Scope: {}", args.scope);
    println!("Governance: {}", args.governance);
    println!("Transparency: {}", args.transparency);
    println!("Modeling parameters under governance active.");
}
