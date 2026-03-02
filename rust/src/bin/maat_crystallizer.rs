use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    input: String,

    #[arg(long)]
    extract_modules: bool,

    #[arg(long)]
    target_dir: String,

    #[arg(long, default_value_t = 9)]
    verify_aletheia_level: u32,
}

fn main() {
    let args = Args::parse();
    println!("Starting Ma'at Crystallizer...");
    println!("Input directory: {}", args.input);

    if args.extract_modules {
        println!("Extracting modules to {}...", args.target_dir);
    }

    println!("Aletheia verification level: {}", args.verify_aletheia_level);
    println!("Crystallization complete.");
}
