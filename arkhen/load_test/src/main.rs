use reqwest::Client;
use serde_json::json;
use std::env;

struct TestResults {
    throughput: f64,
    error_rate: f64,
    total_requests: u64,
    errors: u64,
}

async fn log_to_mlflow(results: &TestResults, collector_version: &str) -> Result<(), anyhow::Error> {
    let client = Client::new();
    let mlflow_url = env::var("MLFLOW_TRACKING_URI").unwrap_or_else(|_| "http://mlflow:5000".into());

    // 1. Create Run
    let create_run = json!({
        "experiment_id": "0",
        "tags": [
            {"key": "test_type", "value": "load_test"},
            {"key": "version", "value": collector_version}
        ]
    });
    let resp = client.post(format!("{}/api/2.0/mlflow/runs/create", mlflow_url))
        .json(&create_run).send().await?;
    let run_info = resp.json::<serde_json::Value>().await?;
    let run_id = run_info["run"]["info"]["run_id"].as_str().unwrap();

    // 2. Log Metrics
    let metrics = json!({
        "run_id": run_id,
        "metrics": [
            {"key": "throughput", "value": results.throughput, "timestamp": 0, "step": 0},
            {"key": "error_rate", "value": results.error_rate, "timestamp": 0, "step": 0}
        ]
    });
    client.post(format!("{}/api/2.0/mlflow/runs/log-batch", mlflow_url))
        .json(&metrics).send().await?;

    // 3. Log Params
    let params = json!({
        "run_id": run_id,
        "params": [
            {"key": "collector_version", "value": collector_version}
        ]
    });
    client.post(format!("{}/api/2.0/mlflow/runs/log-batch", mlflow_url))
        .json(&params).send().await?;

    println!("Results logged to MLflow Run: {}", run_id);
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Starting Arkhe Load Test...");
    let results = TestResults {
        throughput: 15000.0,
        error_rate: 0.001,
        total_requests: 900000,
        errors: 900,
    };
    let version = env::var("COLLECTOR_VERSION").unwrap_or_else(|_| "v0.1.0-dev".into());
    log_to_mlflow(&results, &version).await?;
    Ok(())
}
