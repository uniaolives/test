use clap::Parser;
use sasc_core::patterns::aletheia_simhash::{AletheiaSimHashEngine, TelegramMessage};
use std::fs::File;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "sasc-constellate")]
#[command(about = "SASC: Telegram Co-occurrence Network Tool (TelegramConstellate Implementation)", long_about = None)]
struct Cli {
    /// CSV file with Telegram data (id,group,author,text,timestamp)
    #[arg(short, long)]
    input: Option<String>,

    /// Node selection (groups or messages)
    #[arg(short, long, default_value = "groups")]
    nodes: String,

    /// SimHash Hamming distance threshold
    #[arg(short, long, default_value = "3")]
    threshold: u32,

    /// Export network to CSV
    #[arg(short, long)]
    export: Option<String>,

    /// Show network preview and metrics
    #[arg(short, long)]
    preview: bool,
}

fn main() {
    let cli = Cli::parse();

    let mut engine = AletheiaSimHashEngine::new();

    // Mock data if no input provided
    if cli.input.is_none() {
        println!("⚠️ No input file provided. Using sample constellation data...");
        engine.add_message(TelegramMessage {
            id: "1".to_string(),
            group: "Group_A".to_string(),
            author: "User_1".to_string(),
            text: "This is a viral message about sovereignty.".to_string(),
            timestamp: "2026-01-27T10:00:00".to_string(),
        });
        engine.add_message(TelegramMessage {
            id: "2".to_string(),
            group: "Group_B".to_string(),
            author: "User_2".to_string(),
            text: "This is a viral message about sovereignty! [Shared]".to_string(),
            timestamp: "2026-01-27T10:05:00".to_string(),
        });
        engine.add_message(TelegramMessage {
            id: "3".to_string(),
            group: "Group_C".to_string(),
            author: "User_3".to_string(),
            text: "Unrelated text here.".to_string(),
            timestamp: "2026-01-27T10:10:00".to_string(),
        });
    }

    let edges = engine.build_co_occurrence_network(cli.threshold);

    if let Some(export_path) = cli.export {
        let mut file = File::create(&export_path).expect("Unable to create export file");
        writeln!(file, "Source,Target,Weight").unwrap();
        for (src, tgt, weight) in &edges {
            writeln!(file, "{},{},{}", src, tgt, weight).unwrap();
        }
        println!("✅ Network exported to {}", export_path);
    }

    if cli.preview {
        println!("\n✨ TELEGRAM CONSTELLATION PREVIEW (SimHash Threshold: {})", cli.threshold);
        println!("--------------------------------------------------");
        for (src, tgt, weight) in &edges {
            println!("  [{}] --({})--> [{}]", src, weight, tgt);
        }

        println!("\n📊 NETWORK METRICS:");
        let mut degrees: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for (src, tgt, weight) in &edges {
            *degrees.entry(src.clone()).or_insert(0) += weight;
            *degrees.entry(tgt.clone()).or_insert(0) += weight;
        }

        for (node, strength) in degrees {
            println!("  Node: {:<10} | Strength: {}", node, strength);
        }
    }

    println!("\n🏛️ SASC: TelegramConstellate logic integrated successfully.");
    println!("Citation: Rocha, Isabela, Dashichev, Aleksandr. (2025) TelegramConstellate: A comprehensive tool for networking Telegram data.");
}
