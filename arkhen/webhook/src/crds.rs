use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use kube::CustomResource;

#[derive(CustomResource, Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[kube(group = "arkhe.quantum", version = "v1alpha1", kind = "QuantumManifoldNode", namespaced)]
#[serde(rename_all = "camelCase")]
pub struct QuantumManifoldNodeSpec {
    pub node_id: String,
    pub node_type: String,
    pub location: Option<Location>,
    pub quantum_capabilities: Option<QuantumCapabilities>,
    pub desired_phi: f64,
    pub spin: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub region: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct QuantumCapabilities {
    pub has_qkd: bool,
    pub has_entanglement_source: bool,
    pub max_qubits: Option<i32>,
}

#[derive(CustomResource, Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[kube(group = "arkhe.quantum", version = "v1alpha1", kind = "QuantumChannel", namespaced)]
#[serde(rename_all = "camelCase")]
pub struct QuantumChannelSpec {
    pub source_node: String,
    pub target_node: String,
    pub channel_type: String,
    pub fiber_spec: Option<FiberSpec>,
    pub qkd_protocol: Option<String>,
    pub key_rate: f64,
    pub error_threshold: Option<f64>,
    pub entanglement_consumed: f64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FiberSpec {
    pub length_km: f64,
    pub attenuation: f64,
    pub has_amplifiers: bool,
}
