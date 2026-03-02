use kube::{Client, Api, api::{Patch, PatchParams}, ResourceExt};
use kube::runtime::{controller::Action, Controller};
use std::sync::Arc;
use tokio::time::Duration;
use arkhe_crds::{QuantumManifoldNode, QuantumChannel};
use futures::StreamExt;

struct Context {
    client: Client,
}

async fn reconcile_node(node: Arc<QuantumManifoldNode>, ctx: Arc<Context>) -> Result<Action, kube::Error> {
    let ns = node.namespace().unwrap_or_else(|| "default".to_string());
    let name = node.name_any();
    println!("Reconciling QuantumManifoldNode: {}/{}", ns, name);

    // Simulated status update
    let api: Api<QuantumManifoldNode> = Api::namespaced(ctx.client.clone(), &ns);
    let status = serde_json::json!({
        "status": {
            "current_phi": node.spec.desired_phi,
            "entropy": (1.0 - node.spec.desired_phi).abs()
        }
    });

    // api.patch_status(&name, &PatchParams::default(), &Patch::Merge(&status)).await?;

    Ok(Action::requeue(Duration::from_secs(60)))
}

async fn reconcile_channel(channel: Arc<QuantumChannel>, ctx: Arc<Context>) -> Result<Action, kube::Error> {
    let ns = channel.namespace().unwrap_or_else(|| "default".to_string());
    println!("Reconciling QuantumChannel: {}/{}", ns, channel.name_any());
    Ok(Action::requeue(Duration::from_secs(300)))
}

fn error_policy(_resource: Arc<QuantumManifoldNode>, _error: &kube::Error, _ctx: Arc<Context>) -> Action {
    Action::requeue(Duration::from_secs(30))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Starting Arkhe(n) Reconciliation Controller (Ω+211)...");
    let client = Client::try_default().await?;
    let context = Arc::new(Context { client: client.clone() });

    let nodes = Api::<QuantumManifoldNode>::all(client.clone());
    let channels = Api::<QuantumChannel>::all(client.clone());

    let node_controller = Controller::new(nodes, Default::default())
        .run(reconcile_node, error_policy, context.clone())
        .for_each(|_| async {});

    tokio::select! {
        _ = node_controller => println!("Node controller exited"),
    }

    Ok(())
}
