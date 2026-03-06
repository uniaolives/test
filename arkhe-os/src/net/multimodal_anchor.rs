use tokio::sync::broadcast;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConnectivityLayer {
    BluetoothLE,
    WiFiLocal,
    Cellular5G,
    LoRa,
    SDRDirect,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultimodalCoherencePacket {
    pub layer: ConnectivityLayer,
    pub phi_q_local: f64,
    pub timestamp: u64,
    pub latency_ms: f64,
    pub signal_quality: f64,
    pub bio_signature: Option<String>,
    pub zpf_anomaly: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorUpdate {
    pub device_id: String,
    pub timestamp: i64,
    pub sensor_fusion: serde_json::Value,
    pub witness_signature: String,
}
