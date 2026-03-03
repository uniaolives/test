use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tracing::info;

#[derive(Debug, Serialize, Deserialize)]
struct HandoverRequest {
    context_summary: String,
    priority: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct HandoverResponse {
    status: String,
    handover_id: String,
    phi_remote: f64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/handover", post(handle_handover));

    let addr = SocketAddr::from(([0, 0, 0, 0], 7473));
    info!("Mock CERN Node escutando em {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> &'static str {
    "OK"
}

async fn handle_handover(Json(payload): Json<HandoverRequest>) -> Json<HandoverResponse> {
    info!("Recebido handover: {:?}", payload);
    Json(HandoverResponse {
        status: "synced".to_string(),
        handover_id: "test-handover-uuid".to_string(),
        phi_remote: 0.618,
    })
}
