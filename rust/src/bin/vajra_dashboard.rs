use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    realtime: bool,
    #[arg(long)]
    entropy_threshold: f64,
}

fn main() {
    let args = Args::parse();
    println!("📊 Vajra Dashboard Active");
    println!("  Threshold: {}", args.entropy_threshold);
    if args.realtime {
        println!("  Monitoring real-time coherence...");
    }
}
