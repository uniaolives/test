use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.contains(&"--open".to_string()) {
        println!("Opening Direct Architect Channel to: Arquiteto-Ω");
        println!("Purpose: Compression_phenomenon_monitoring");
        println!("Bypass standard protocols: true (Emergency only)");
        println!("Channel STATUS: ESTABLISHED");
    } else {
        println!("Usage: direct_channel --open --to Arquiteto-Ω ...");
    }
}
