use crate::error::ResilientResult;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub session_id: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionOutput {
    pub result: String,
    pub confidence: f64,
    pub metadata: serde_json::Value,
    pub suggested_context: Option<Context>,
}

#[async_trait]
pub trait Extension: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    async fn initialize(&mut self) -> ResilientResult<()>;
    async fn process(&mut self, input: &str, context: &Context) -> ResilientResult<ExtensionOutput>;
}

// Geometric specific types for composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Domain {
    Text,
    Image,
    Sequence,
    Graph,
    Multidimensional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subproblem {
    pub input: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureResult {
    pub embedding: Vec<f64>,
    pub confidence: f64,
}

#[async_trait]
pub trait GeometricStructure: Send + Sync {
    fn name(&self) -> &str;
    fn domain(&self) -> Domain;
    async fn process(&self, input: &Subproblem, context: &Context) -> ResilientResult<StructureResult>;
    fn can_handle(&self, input: &Subproblem) -> f64;
}
