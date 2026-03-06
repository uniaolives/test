use tokio::sync::mpsc;
use std::sync::Arc;
use crate::telemetry::GlobalState;

pub enum NetEvent {
    Update { timestamp: u64, layer: String, rssi: f64 },
}

pub async fn start_multimodal_stack(_tx: mpsc::Sender<NetEvent>, _state: Arc<GlobalState>) {
    println!("[NET] Starting Multimodal Stack (BLE/WiFi/5G)...");
}
