use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSample {
    pub source: String,
    pub timestamp: DateTime<Utc>,
    pub shard_id: String,
    pub metric_name: String,
    pub value: f64,
}

pub struct TelemetryCollector {
    tx: flume::Sender<MetricSample>,
}

impl TelemetryCollector {
    pub fn new(tx: flume::Sender<MetricSample>) -> Self {
        Self { tx }
    }

    pub async fn run(&self) {
        println!("Telemetry Collector active.");
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let (tx, _rx) = flume::bounded(10000);
    let collector = TelemetryCollector::new(tx);
    collector.run().await;
    Ok(())
}
