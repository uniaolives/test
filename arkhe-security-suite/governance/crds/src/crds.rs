use kube::CustomResource;
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

#[derive(CustomResource, Serialize, Deserialize, Default, Clone, Debug, JsonSchema)]
#[kube(group = "arkhe.quantum", version = "v1", kind = "QuantumManifoldNode", namespaced)]
#[kube(status = "QuantumManifoldNodeStatus")]
pub struct QuantumManifoldNodeSpec {
    pub desired_phi: f64,
    pub quantum_capabilities: Option<QuantumCapabilities>,
}

#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema)]
pub struct QuantumCapabilities {
    pub has_qkd: bool,
    pub has_memory: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Default)]
pub struct QuantumManifoldNodeStatus {
    pub current_phi: f64,
    pub entropy: f64,
}

#[derive(CustomResource, Serialize, Deserialize, Default, Clone, Debug, JsonSchema)]
#[kube(group = "arkhe.quantum", version = "v1", kind = "QuantumChannel", namespaced)]
pub struct QuantumChannelSpec {
    pub source_node: String,
    pub target_node: String,
    pub channel_type: String, // "darkFiber", "satelliteQKD"
    pub key_rate: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Default)]
pub struct QuantumChannelStatus {
    pub active: bool,
}
