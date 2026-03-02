use reqwest::Client;
use tokio::time::{sleep, Duration, Instant};
use std::sync::Arc;
use tokio::sync::Semaphore;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "http://localhost:8080/api/handovers")]
    target: String,

    #[arg(short, long, default_value_t = 100)]
    concurrency: usize,

    #[arg(short, long, default_value_t = 1000)]
    rps: u64,

    #[arg(short, long, default_value_t = 10)]
    duration: u64,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let client = Arc::new(Client::new());
    let semaphore = Arc::new(Semaphore::new(args.concurrency));
    let interval = Duration::from_secs_f64(1.0 / (args.rps as f64));
    let start_time = Instant::now();
    let test_duration = Duration::from_secs(args.duration);

    println!("🚀 ARKHE(n) Load Test – Protocol Ω+222");
    println!("Target: {}", args.target);
    println!("Concurrency: {}, RPS: {}, Duration: {}s", args.concurrency, args.rps, args.duration);

    while start_time.elapsed() < test_duration {
        let permit = match semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                sleep(Duration::from_millis(1)).await;
                continue;
            }
        };

        let client = client.clone();
        let target = args.target.clone();

        tokio::spawn(async move {
            let payload = serde_json::json!({
                "emitter_id": 1,
                "receiver_id": 2,
                "entropy_cost": 0.05,
                "magic": "ARKH"
            });
            let _ = client.post(&target).json(&payload).send().await;
            drop(permit);
        });

        sleep(interval).await;
    }

    println!("✅ Load test completed.");
}
