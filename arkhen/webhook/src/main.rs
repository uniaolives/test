mod crds;
mod validator;

use actix_web::{post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use kube::core::admission::{AdmissionReview};
use kube::core::DynamicObject;
use kube::Client;
use std::fs;
use tracing::{info, warn};

#[post("/validate")]
async fn validate(client: web::Data<Client>, body: web::Bytes) -> impl Responder {
    let admission_review: AdmissionReview<DynamicObject> = match serde_json::from_slice(&body) {
        Ok(ar) => ar,
        Err(e) => {
            warn!("Failed to parse admission review: {}", e);
            return HttpResponse::BadRequest().body(format!("Invalid JSON: {}", e));
        }
    };

    let request = match admission_review.request {
        Some(req) => req,
        None => {
            return HttpResponse::BadRequest().body("Missing request");
        }
    };

    let response = validator::validate_admission_review(&request, &client).await;

    let review_response = AdmissionReview::<DynamicObject> {
        types: admission_review.types,
        request: None,
        response: Some(response),
    };

    HttpResponse::Ok().json(review_response)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // tracing_subscriber::fmt::init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "8443".to_string());
    let cert_file = std::env::var("TLS_CERT").unwrap_or_else(|_| "/tls/tls.crt".to_string());
    let key_file = std::env::var("TLS_KEY").unwrap_or_else(|_| "/tls/tls.key".to_string());

    info!("Starting Arkhe validating webhook on port {}", port);

    // TLS is required for production, but we provide a check-friendly mode
    let cert_data = fs::read(&cert_file).unwrap_or_default();
    let key_data = fs::read(&key_file).unwrap_or_default();

    let client = Client::try_default().await.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    if cert_data.is_empty() || key_data.is_empty() {
        warn!("TLS certificates not found. Webhook will start in HTTP mode (UNSAFE).");
        return HttpServer::new(move || {
            App::new()
                .app_data(web::Data::new(client.clone()))
                .service(validate)
        })
        .bind(format!("0.0.0.0:{}", port))?
        .run()
        .await;
    }

    let certs = rustls_pemfile::certs(&mut cert_data.as_slice())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let mut keys = rustls_pemfile::pkcs8_private_keys(&mut key_data.as_slice())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let tls_config = rustls::ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(
            certs.into_iter().map(rustls::Certificate).collect(),
            rustls::PrivateKey(keys.remove(0))
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(client.clone()))
            .service(validate)
    })
    .bind_rustls(format!("0.0.0.0:{}", port), tls_config)?
    .run()
    .await
}
