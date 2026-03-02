//! neuraxon.rs - Rust Bridge between neural dynamics and Arkhe(n) handovers (Ω+195)
//! Integrates biological trinary logic into the high-performance core.

use std::sync::Arc;

pub enum NeuraxonState {
    Excitatory,
    Inhibitory,
    Modulatory,
}

pub struct Neuraxon {
    pub id: usize,
    pub potential: f64,
}

impl Neuraxon {
    pub fn output_state(&self) -> NeuraxonState {
        if self.potential > 0.5 {
            NeuraxonState::Excitatory
        } else if self.potential < -0.5 {
            NeuraxonState::Inhibitory
        } else {
            NeuraxonState::Modulatory
        }
    }
}

pub struct NeuraxonAdapter {
    pub neurons: Vec<Neuraxon>,
    pub node_id: usize,
}

impl NeuraxonAdapter {
    pub fn new(node_id: usize, num_neurons: usize) -> Self {
        let mut neurons = Vec::with_capacity(num_neurons);
        for id in 0..num_neurons {
            neurons.push(Neuraxon { id, potential: 0.0 });
        }
        Self { neurons, node_id }
    }

    /// Converts neural state to constitutional handover
    pub fn process_neuron(&self, neuron_id: usize) -> Option<String> {
        let neuron = &self.neurons[neuron_id];
        match neuron.output_state() {
            NeuraxonState::Excitatory => Some(format!("Handover::Excitatory from node {}", self.node_id)),
            NeuraxonState::Inhibitory => Some(format!("Handover::Inhibitory (Veto) from node {}", self.node_id)),
            NeuraxonState::Modulatory => None, // Latent modulation
        }
    }
}
