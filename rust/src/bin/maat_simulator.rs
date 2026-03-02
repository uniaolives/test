use clap::Parser;
use sasc_core::maat::scenarios::tumor_microenvironment::TumorMicroenvironment;
use sasc_core::maat::scenarios::network_congestion::DDoSResilienceTest;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    scenario: String,

    #[arg(long, default_value_t = 86400)]
    duration: u64,

    #[arg(long, default_value_t = 1000)]
    swarm_size: usize,

    #[arg(long, default_value_t = 0.8)]
    collagen_density: f64,

    #[arg(long, default_value_t = 0.95)]
    ubuntu_cohesion: f64,

    #[arg(long)]
    output: String,

    #[arg(long, default_value_t = 0.40)]
    byzantine_ratio: f64,

    #[arg(long)]
    attack_vector: Option<String>,

    #[arg(long)]
    screw_propulsion: Option<String>,

    #[arg(long)]
    ubuntu_consensus: Option<String>,
}

fn main() {
    let args = Args::parse();
    println!("Starting Ma'at Simulator...");
    println!("Scenario: {}", args.scenario);

    match args.scenario.as_str() {
        "tumor-navigation" => {
            let seed = [0u8; 32];
            let mut sim = TumorMicroenvironment::initialize_with_seed(&seed, args.swarm_size);
            let metrics = sim.run_simulation(args.duration as f64 / 3600.0);
            println!("Simulation complete. Stuck rate: {:.4}, Delivery: {:.2}", metrics.stuck_rate, metrics.drug_delivered);
        }
        "network-congestion" => {
            println!("Running network-congestion scenario...");
            let mut test = DDoSResilienceTest::initialize(args.swarm_size);

            let attack_vector = match args.attack_vector.as_deref() {
                Some("syn_flood") => sasc_core::maat::scenarios::network_congestion::AttackVector::SynFlood,
                _ => sasc_core::maat::scenarios::network_congestion::AttackVector::SynFlood,
            };

            test.configure_attack(args.byzantine_ratio, attack_vector);

            println!("Running baseline test...");
            let test_dur = Duration::from_millis(args.duration * 100);
            let baseline = test.run_baseline_test(test_dur); // Scaled down for demo
            println!("Baseline Throughput: {:.2}", baseline.throughput);

            println!("Running adaptive test...");
            let adaptive = test.run_adaptive_test(test_dur);
            println!("Adaptive Throughput: {:.2}", adaptive.throughput);

            let report = test.generate_resilience_report();
            println!("Improvement: {:.2}x", report.throughput_improvement);
        }
        _ => println!("Unknown scenario"),
    }

    println!("Results saved to {}", args.output);
}
