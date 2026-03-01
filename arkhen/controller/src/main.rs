use kube::{
    api::{Api, Patch, PatchParams, ResourceExt},
    client::Client,
    runtime::{controller::Action, Controller, watcher::Config},
    CustomResource,
};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use tokio::time::Duration;
use std::sync::Arc;
use futures::StreamExt;

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

struct Context {
    client: Client,
}

async fn reconcile(node: Arc<QuantumManifoldNode>, ctx: Arc<Context>) -> Result<Action, kube::Error> {
    let namespace = node.namespace().unwrap_or_else(|| "default".into());
    let nodes: Api<QuantumManifoldNode> = Api::namespaced(ctx.client.clone(), &namespace);

    println!("Reconciling Node: {}", node.name_any());

    let status = QuantumManifoldNodeStatus {
        observed_state: Some(ObservedState {
            current_phi: node.spec.desired_phi,
            entropy: 0.1,
        }),
    };

    let patch = Patch::Apply(serde_json::json!({
        "apiVersion": "arkhe.quantum/v1alpha1",
        "kind": "QuantumManifoldNode",
        "status": status
    }));

    nodes.patch_status(&node.name_any(), &PatchParams::apply("arkhe-controller"), &patch).await?;

    Ok(Action::requeue(Duration::from_secs(300)))
}

fn error_policy(_node: Arc<QuantumManifoldNode>, _error: &kube::Error, _ctx: Arc<Context>) -> Action {
    Action::requeue(Duration::from_secs(5))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Arkhe(n) Infrastructure Controller Starting...");
    let client = Client::try_default().await?;
    let nodes = Api::<QuantumManifoldNode>::all(client.clone());
    let context = Arc::new(Context { client });

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
