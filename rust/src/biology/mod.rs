pub type GeneExpressionPrediction = Vec<f64>;

#[derive(Clone, Debug)]
pub struct HeterogeneousGraph {
    pub compromised: bool,
}

impl HeterogeneousGraph {
    pub fn new() -> Self {
        Self { compromised: false }
    }
    pub fn edges(&self) -> Vec<Edge> {
        vec![]
    }
    pub fn provenance_hash(&self) -> [u8; 32] {
        [0u8; 32]
    }
    pub fn source(&self) -> &str {
        "GENESIS_PPI"
    }
    pub fn mark_as_compromised(&mut self) {
        self.compromised = true;
    }
    pub fn is_compromised(&self) -> bool {
        self.compromised
    }
}

pub struct Edge;
impl Edge {
    pub fn metadata(&self) -> Metadata {
        Metadata
    }
}

pub struct Metadata;
impl Metadata {
    pub fn signature(&self) -> [u8; 64] {
        [0u8; 64]
    }
}

#[derive(Debug, Clone)]
pub struct SecureEmbedding;
