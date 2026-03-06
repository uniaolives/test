use tokio::sync::mpsc;
use crate::net::multimodal_anchor::ConnectivityLayer;

pub mod analytics;
pub mod entropy;

pub enum ZPFEvent {
    Spectrum { timestamp: u64, kurtosis: f64, entropy: f64 },
    MultiBand { timestamp_ns: u64, bands: std::collections::HashMap<String, f64> },
pub enum ZPFEvent {
    Spectrum { timestamp: u64, kurtosis: f64, entropy: f64 },
}

pub async fn start_zpf_pipeline(tx: mpsc::Sender<ZPFEvent>) {
    println!("[SENSORS] Starting ZPF Pipeline (ZMQ sub)...");
    let context = zmq::Context::new();
    let subscriber = context.socket(zmq::SUB).unwrap();

    // Connect to GNU Radio / pySDR
    if let Ok(_) = subscriber.connect("tcp://127.0.0.1:5556") {
        subscriber.set_subscribe(b"").unwrap();
        tokio::spawn(async move {
            loop {
                if let Ok(msg) = subscriber.recv_msg(0) {
                    if let Ok(data) = serde_json::from_slice::<serde_json::Value>(&msg) {
                        if data["type"] == "zpf_anomaly" {
                            let kurtosis = data["metrics"]["kurtosis"].as_f64().unwrap_or(0.0);
                            let entropy = data["metrics"]["entropy"].as_f64().unwrap_or(0.0);
                            let _ = tx.send(ZPFEvent::Spectrum {
                                timestamp: data["timestamp"].as_u64().unwrap_or(0),
                                kurtosis,
                                entropy,
                            }).await;
                        }
                    }
                }
                tokio::task::yield_now().await;
            }
        });
    }
}
