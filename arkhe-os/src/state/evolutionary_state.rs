// arkhe-os/src/state/evolutionary_state.rs
//! Evolutionary State for Bio-Nodes (Cortex + Whittaker Architecture)
//! IP: Safe Core / Rafael Oliveira

use crate::orb::core::OrbPayload;
use std::collections::VecDeque;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextVector {
    pub embedding: Vec<f64>,
    pub timestamp: i64,
    pub relevance_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub state_version: u64,
    pub summary: String,
    pub importance: f64,
}

pub struct EvolutionaryStateStore {
    pub immediate_context: Vec<ContextVector>,
    pub historical_memory: VecDeque<MemoryRecord>,
    pub context_decay_rate: f64,
    pub state_coherence: f64,
}

impl EvolutionaryStateStore {
    pub fn new() -> Self {
        Self {
            immediate_context: Vec::new(),
            historical_memory: VecDeque::new(),
            context_decay_rate: 0.01,
            state_coherence: 1.0,
        }
    }

    pub fn evolve(&mut self, payload: &OrbPayload) {
        let relevance = self.calculate_relevance(payload);
        let ctx = ContextVector {
            embedding: vec![payload.lambda_2, payload.phi_q, payload.h_value],
            timestamp: payload.origin_time,
            relevance_score: relevance,
        };
        self.immediate_context.push(ctx);
        self.apply_decay();
        self.state_coherence = self.calculate_state_coherence();
    }

    fn calculate_relevance(&self, _payload: &OrbPayload) -> f64 {
        // Importance + Semantic Relevance
        0.8
    }

    fn apply_decay(&mut self) {
        // Randall-Sundrum warp factor e^(-k*t) suppression
        let warp = (-self.context_decay_rate).exp();
        for ctx in &mut self.immediate_context {
            ctx.relevance_score *= warp;
        }
    }

    fn calculate_state_coherence(&self) -> f64 {
        // Stability of the Whittaker manifold
        0.95
    }
}
