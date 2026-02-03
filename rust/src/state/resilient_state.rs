// rust/src/state/resilient_state.rs
use crate::error::{ResilientError, ResilientResult};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilientState {
    pub version: String,
    pub agent_id: String,
    pub instance_id: String,
    pub memory: AgentMemory,
    pub previous_tx_id: Option<String>,
    pub genesis_tx_id: String,
    pub height: u64,
    pub created_at: u64,
    pub updated_at: u64,
    pub state_hash: String,
    pub signature: Option<String>,
    pub torsion_metrics: TorsionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemory {
    pub summary: String,
    pub context_hash: String,
    pub important_decisions: Vec<DecisionRecord>,
    pub knowledge_graph: KnowledgeGraph,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub id: String,
    pub timestamp: u64,
    pub input_hash: String,
    pub output_hash: String,
    pub reasoning_hash: String,
    pub ethical_check: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    pub nodes: Vec<KnowledgeNode>,
    pub edges: Vec<KnowledgeEdge>,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub id: String,
    pub content_hash: String,
    pub node_type: NodeType,
    pub confidence: f32,
    pub source_tx: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    pub from: String,
    pub to: String,
    pub relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Fact,
    Hypothesis,
    Principle,
    Rule,
    Experience,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorsionMetrics {
    pub divergence_from_previous: f64,
    pub ethical_alignment: f64,
    pub logical_consistency: f64,
    pub temporal_coherence: f64,
}

impl ResilientState {
    pub fn new(agent_id: &str) -> Self {
        let ts = timestamp_millis();
        let instance_id = format!("{}-{}", agent_id, ts);

        Self {
            version: "CGE-v1.0".to_string(),
            agent_id: agent_id.to_string(),
            instance_id,
            memory: AgentMemory {
                summary: "".to_string(),
                context_hash: "".to_string(),
                important_decisions: Vec::new(),
                knowledge_graph: KnowledgeGraph {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                    version: "1.0".to_string(),
                },
                timestamp: ts,
            },
            previous_tx_id: None,
            genesis_tx_id: "".to_string(),
            height: 0,
            created_at: ts,
            updated_at: ts,
            state_hash: "".to_string(),
            signature: None,
            torsion_metrics: TorsionMetrics {
                divergence_from_previous: 0.0,
                ethical_alignment: 1.0,
                logical_consistency: 1.0,
                temporal_coherence: 1.0,
            },
        }
    }

    pub fn prepare_for_checkpoint(&mut self, previous_tx_id: Option<String>) -> ResilientResult<()> {
        self.previous_tx_id = previous_tx_id;
        self.height += 1;
        self.updated_at = timestamp_millis();
        self.state_hash = self.calculate_hash()?;
        Ok(())
    }

    pub fn calculate_hash(&self) -> ResilientResult<String> {
        let serialized = serde_json::to_vec(self)
            .map_err(|e| ResilientError::StateValidation(format!("Serialization failed: {}", e)))?;
        let hash = blake3::hash(&serialized);
        Ok(hash.to_hex().to_string())
    }

    pub fn estimate_size(&self) -> ResilientResult<usize> {
        let serialized = serde_json::to_vec(self)
            .map_err(|e| ResilientError::StateValidation(format!("Serialization failed: {}", e)))?;
        Ok(serialized.len())
    }
}

fn timestamp_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
