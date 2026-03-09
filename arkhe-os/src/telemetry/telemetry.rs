use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::sync::RwLock;
use warp::Filter;
use futures_util::{StreamExt, SinkExt};
use serde_json;

pub enum BioEvent {
    Telemetry { timestamp: u64, accel: f64, mag: f64, hrv: f64 },
}

pub struct GlobalState {
    pub phi_q: RwLock<f64>,
    pub coherence_history: RwLock<Vec<f64>>,
}

pub async fn start_bio_server(tx: mpsc::Sender<BioEvent>, state: Arc<GlobalState>) {
    println!("[TELEMETRY] Starting Bio Server (WSS 3030)...");

    let bio_route = warp::path("anchor")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let tx_inner = tx.clone();
            let state_inner = state.clone();
            ws.on_upgrade(move |mut websocket| async move {
                while let Some(Ok(msg)) = websocket.next().await {
                    if let Ok(text) = msg.to_str() {
                        if let Ok(data) = serde_json::from_str::<serde_json::Value>(text) {
                            let accel = data["accel_variance"].as_f64().unwrap_or(0.0);
                            let mag = data["mag_field"].as_f64().unwrap_or(0.0);
                            let _ = tx_inner.send(BioEvent::Telemetry {
                                timestamp: data["timestamp"].as_u64().unwrap_or(0),
                                accel,
                                mag,
                                hrv: 0.0,
                            }).await;

                            // Send back current phi_q
                            let phi = *state_inner.phi_q.read().await;
                            let response = serde_json::json!({
                                "global_phi_q": phi,
                                "recommended_state": if phi > 4.64 { "high_coherence" } else { "normal" }
                            });
                            let _ = websocket.send(warp::ws::Message::text(response.to_string())).await;
                        }
                    }
                }
            })
        });

    tokio::spawn(async move {
        warp::serve(bio_route).run(([0, 0, 0, 0], 3030)).await;
    });
}
