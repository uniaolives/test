use actix_web::{post, web, App, HttpRequest, HttpServer, Responder, HttpResponse};
use kube::{Client, Api, ResourceExt, api::ListParams};
use serde_json::Value;
use arkhe_crds::{QuantumManifoldNode, QuantumChannel, QuantumChannelSpec};

struct AppState {
    kube_client: Client,
}

async fn validate_global_entropy(spec: &arkhe_crds::QuantumManifoldNodeSpec, client: &Client, ns: &str) -> Result<(), String> {
    let api: Api<QuantumManifoldNode> = Api::namespaced(client.clone(), ns);
    let nodes = api.list(&ListParams::default()).await.map_err(|e| e.to_string())?;
    let total_phi: f64 = nodes.items.iter().map(|n| n.spec.desired_phi).sum();
    if total_phi + spec.desired_phi > 100.0 {
        return Err("Global phi/entropy limit exceeded (max 100)".to_string());
    }
    Ok(())
}

async fn validate_channel_nodes(spec: &QuantumChannelSpec, client: &Client, ns: &str) -> Result<(), String> {
    let api: Api<QuantumManifoldNode> = Api::namespaced(client.clone(), ns);
    if api.get(&spec.source_node).await.is_err() {
        return Err(format!("Source node '{}' not found", spec.source_node));
    }
    if api.get(&spec.target_node).await.is_err() {
        return Err(format!("Target node '{}' not found", spec.target_node));
    }
    Ok(())
}

#[post("/validate")]
async fn validate(
    req: HttpRequest,
    body: web::Json<Value>,
    data: web::Data<AppState>,
) -> impl Responder {
    let request = &body["request"];
    let kind = request["kind"]["kind"].as_str().unwrap_or("");
    let ns = request["namespace"].as_str().unwrap_or("default");
    let object = &request["object"];

    let mut allowed = true;
    let mut message = "Allowed by Arkhe(n) Governance".to_string();

    match kind {
        "QuantumManifoldNode" => {
            if let Ok(spec) = serde_json::from_value(object["spec"].clone()) {
                if let Err(e) = validate_global_entropy(&spec, &data.kube_client, ns).await {
                    allowed = false;
                    message = e;
                }
            }
        }
        "QuantumChannel" => {
            if let Ok(spec) = serde_json::from_value(object["spec"].clone()) {
                if let Err(e) = validate_channel_nodes(&spec, &data.kube_client, ns).await {
                    allowed = false;
                    message = e;
                }
            }
        }
        _ => {}
    }

    HttpResponse::Ok().json(serde_json::json!({
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": request["uid"].as_str().unwrap_or(""),
            "allowed": allowed,
            "status": {
                "message": message
            }
        }
    }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Starting Arkhe(n) Validating Webhook (Ω+211)...");
    // In a real environment, we would load SSL and bind to 443
    HttpServer::new(|| {
        App::new()
            .service(validate)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
