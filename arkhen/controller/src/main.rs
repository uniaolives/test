use kube::{
    api::{Api, Patch, PatchParams, ResourceExt, ListParams, DeleteParams},
    client::Client,
    runtime::{controller::Action, Controller, watcher::Config},
    CustomResource,
};
use k8s_openapi::api::core::v1::Node;
use k8s_openapi::api::policy::v1::Eviction;
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use tokio::time::Duration;
use std::sync::Arc;
use futures::StreamExt;
use reqwest::Client as HttpClient;
use chrono::Utc;

#[derive(CustomResource, Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[kube(group = "arkhe.quantum", version = "v1alpha1", kind = "QuantumManifoldNode", namespaced)]
#[kube(status = "QuantumManifoldNodeStatus")]
pub struct QuantumManifoldNodeSpec {
    pub node_id: String,
    pub desired_phi: f64,
    pub spin: Option<f64>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
pub struct QuantumManifoldNodeStatus {
    pub observed_state: Option<ObservedState>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
pub struct ObservedState {
    pub current_phi: f64,
    pub entropy: f64,
}

#[derive(CustomResource, Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[kube(group = "arkhe.quantum", version = "v1alpha1", kind = "EntropyPrediction", namespaced)]
#[kube(status = "EntropyPredictionStatus")]
pub struct EntropyPredictionSpec {
    pub target_node: String,
    pub horizon: Option<i32>,
    pub threshold: Option<f64>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EntropyPredictionStatus {
    pub predictions: Vec<PredictionPoint>,
    pub last_update: chrono::DateTime<Utc>,
    pub model_version: String,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PredictionPoint {
    pub timestamp: chrono::DateTime<Utc>,
    pub predicted_entropy: f64,
    pub confidence: f64,
    pub failure_probability: f64,
}

#[derive(Deserialize)]
struct PredictionResponse {
    pub failure_probability: f64,
}

struct Context {
    client: Client,
    http: HttpClient,
    predictor_url: String,
}

async fn drain_node(client: Client, node_name: &str) -> Result<(), anyhow::Error> {
    println!("Initiating aggressive auto-healing (drain) for node {}", node_name);
    let nodes: Api<Node> = Api::all(client.clone());

    // Cordon
    let node = nodes.get(node_name).await?;
    let mut cordon_node = node.clone();
    cordon_node.spec.as_mut().unwrap().unschedulable = Some(true);
    nodes.patch(node_name, &PatchParams::apply("arkhe-controller"), &Patch::Apply(cordon_node)).await?;

    // Pod Eviction simplified for bootstrap
    println!("Node {} cordoned. Pod eviction would follow in production.", node_name);
    Ok(())
}

async fn reconcile(node: Arc<QuantumManifoldNode>, ctx: Arc<Context>) -> Result<Action, kube::Error> {
    let namespace = node.namespace().unwrap_or_else(|| "default".into());
    let nodes: Api<QuantumManifoldNode> = Api::namespaced(ctx.client.clone(), &namespace);
    let preds: Api<EntropyPrediction> = Api::namespaced(ctx.client.clone(), &namespace);

    // 1. Prediction Loop
    let url = format!("{}/predict/{}", ctx.predictor_url, node.name_any());
    if let Ok(resp) = ctx.http.get(&url).send().await {
        if let Ok(data) = resp.json::<PredictionResponse>().await {
             let pred_name = format!("pred-{}", node.name_any());
             let prob = data.failure_probability;

             // Auto-heal if probability > 0.9
             if prob > 0.9 {
                 let _ = drain_node(ctx.client.clone(), &node.name_any()).await;
             }

             let pred_status = EntropyPredictionStatus {
                 predictions: vec![PredictionPoint {
                     timestamp: Utc::now(),
                     predicted_entropy: 0.5,
                     confidence: 0.9,
                     failure_probability: prob,
                 }],
                 last_update: Utc::now(),
                 model_version: "lstm-v2-autoheal".into(),
             };

             let pred_patch = Patch::Apply(serde_json::json!({
                 "apiVersion": "arkhe.quantum/v1alpha1",
                 "kind": "EntropyPrediction",
                 "spec": { "targetNode": node.name_any() },
                 "status": pred_status
             }));
             let _ = preds.patch(&pred_name, &PatchParams::apply("arkhe-controller").force(), &pred_patch).await;
        }
    }

    Ok(Action::requeue(Duration::from_secs(300)))
}

fn error_policy(_node: Arc<QuantumManifoldNode>, _error: &kube::Error, _ctx: Arc<Context>) -> Action {
    Action::requeue(Duration::from_secs(5))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Arkhe(n) Controller with Auto-Healing Starting...");
    let client = Client::try_default().await?;
    let nodes = Api::<QuantumManifoldNode>::all(client.clone());
    let context = Arc::new(Context {
        client,
        http: HttpClient::new(),
        predictor_url: std::env::var("PREDICTOR_URL").unwrap_or_else(|_| "http://predictor:5000".into()),
    });

    Controller::new(nodes, Config::default())
        .run(reconcile, error_policy, context)
        .for_each(|res| async move {
            match res {
                Ok(o) => println!("reconciled {:?}", o),
                Err(e) => println!("reconcile failed: {:?}", e),
            }
        })
        .await;
    Ok(())
}
