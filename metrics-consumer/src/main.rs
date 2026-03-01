use serde::{Deserialize, Serialize};
use rdkafka::consumer::{Consumer, StreamConsumer, CommitMode};
use rdkafka::config::ClientConfig;
use rdkafka::message::Message;

#[derive(Serialize, Deserialize, Debug)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub labels: std::collections::HashMap<String, String>,
}

async fn _consume_metrics() -> anyhow::Result<()> {
    let consumer: StreamConsumer = ClientConfig::new()
        .set("group.id", "arkhe-metrics-group")
        .set("bootstrap.servers", "kafka:9092")
        .set("enable.partition.eof", "false")
        .set("session.timeout.ms", "6000")
        .set("enable.auto.commit", "true")
        .create()?;

    consumer.subscribe(&["arkhe-metrics"])?;

    println!("Monitoring Kafka topic 'arkhe-metrics'...");

    loop {
        match consumer.recv().await {
            Err(e) => println!("Kafka error: {}", e),
            Ok(m) => {
                let payload = match m.payload_view::<str>() {
                    None => "",
                    Some(Ok(s)) => s,
                    Some(Err(e)) => {
                        println!("Error decoding message payload: {:?}", e);
                        ""
                    }
                };

                if let Ok(metric) = serde_json::from_str::<Metric>(payload) {
                    println!("Received metric: {} = {}", metric.name, metric.value);
                }
                consumer.commit_message(&m, CommitMode::Async)?;
            }
        };
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Arkhe(n) Metrics Consumer starting...");
    Ok(())
}
