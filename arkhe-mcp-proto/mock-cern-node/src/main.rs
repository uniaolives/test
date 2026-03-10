use axum::{
    routing::{get, post},
    Json, Router,
};
use std::net::SocketAddr;
use tracing::info;
use arkhe_api_rust::objects::com::palantir::arkhe::api::{
    HandoverRequest, HandoverResponse, Orb, CoherenceMetric
};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/handover/sync", post(handle_handover))
        .route("/temporal/emit", post(handle_emit));

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
    Json(HandoverResponse::builder()
        .status("synced")
        .handover_id("test-handover-uuid")
        .phi_remote(0.618)
        .build())
}

async fn handle_emit(Json(payload): Json<Orb>) -> Json<CoherenceMetric> {
    info!("Recebido orb: {:?}", payload);
    Json(CoherenceMetric::builder()
        .global_r(0.95)
        .node_lambda(payload.lambda2())
        .stable(true)
        .build())
}
