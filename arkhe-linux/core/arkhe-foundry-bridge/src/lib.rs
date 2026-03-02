use serde::{Serialize, Deserialize};
use arkhe_quantum::{Handover, HandoverType};
use uuid::Uuid;

pub mod arkhe_grpc {
    tonic::include_proto!("arkhe");
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CognitiveNode {
    pub node_id: String,
    pub phi: f64,
    pub entropy: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QuantumLink {
    pub source_id: String,
    pub target_id: String,
    pub correlation: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HandoverLog {
    pub handover_id: String,
    pub payload: String,
}

pub struct FoundryBridge {
    pub api_endpoint: String,
}

impl FoundryBridge {
    pub fn new(endpoint: &str) -> Self {
        Self { api_endpoint: endpoint.to_string() }
    }

    pub async fn update_node_phi(&self, node: &CognitiveNode) -> anyhow::Result<()> {
        let mut client = arkhe_grpc::arkhe_service_client::ArkheServiceClient::connect("http://localhost:50051").await?;

        let request = tonic::Request::new(arkhe_grpc::OntologyRequest {
            object_id: node.node_id.clone(),
            object_type: "CognitiveNode".into(),
            payload_json: serde_json::to_string(node)?,
        });

        client.update_ontology(request).await?;
        Ok(())
    }
}
