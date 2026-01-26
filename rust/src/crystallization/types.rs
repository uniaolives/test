use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillSignature(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillId(pub String);

#[derive(Debug, Clone)]
pub struct NativeModule {
    pub name: String,
    pub entry_point: String,
    /// In a real implementation, this would contain WASM bytes or a pointer to a dynamic library
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PureAlgorithm {
    pub graph_representation: String,
}

#[derive(Debug, Clone)]
pub struct ContextualData {
    pub embeddings: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct VectorDBQuery {
    pub query_string: String,
}

pub type Hash = [u8; 32];
