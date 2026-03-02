use reqwest::Client;
use tokio::time::{sleep, Duration};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[tokio::main]
async fn main() {
    let client = Arc::new(Client::new());
    let target = "http://localhost:8080/api/handovers";
    let concurrency = 100;
    let requests_per_second = 1000;

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let interval = Duration::from_secs_f64(1.0 / (requests_per_second as f64));

    println!("Starting load test against {}...", target);

    for _ in 0..100 { // Limit test duration
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let target = target.to_string();

        tokio::spawn(async move {
            let payload = serde_json::json!({
                "emitter_id": 1,
                "receiver_id": 2,
                "entropy_cost": 0.05
            });
            let _ = client.post(&target).json(&payload).send().await;
            drop(permit);
        });

        sleep(interval).await;
    }

    println!("Load test completed.");
}
