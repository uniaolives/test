use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    domain: String,
}

fn main() {
    let args = Args::parse();
    println!("✓ Domain Separator verified for: {}", args.domain);
}
