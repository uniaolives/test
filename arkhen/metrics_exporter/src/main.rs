use prometheus::{register_gauge_vec, GaugeVec, Opts, Encoder, TextEncoder};
use lazy_static::lazy_static;
use actix_web::{get, App, HttpResponse, HttpServer, Responder};

lazy_static! {
    static ref ENTROPY_GAUGE: GaugeVec = register_gauge_vec!(
        Opts::new("arkhe_node_entropy", "Von Neumann entropy of the node"),
        &["node"]
    ).unwrap();
    static ref PHI_GAUGE: GaugeVec = register_gauge_vec!(
        Opts::new("arkhe_node_phi", "Current phi (criticality)"),
        &["node"]
    ).unwrap();
}

#[get("/metrics")]
async fn metrics() -> impl Responder {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();
    HttpResponse::Ok().body(buffer)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new().service(metrics)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
