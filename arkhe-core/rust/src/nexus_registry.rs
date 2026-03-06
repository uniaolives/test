use crate::{ArkheError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NexusRecord {
    pub id: u64,
    pub timestamp: i64,          // época Unix do nó
    pub public_key: Vec<u8>,      // chave para assinar handovers
    pub initial_coherence: f64,
    pub endpoints: Vec<String>,   // endereços de rede
    pub endorsements: Vec<u64>,   // IDs dos nós que atestaram este
    pub active: bool,
}

pub struct NexusRegistry {
    nodes: HashMap<u64, NexusRecord>,
}

impl NexusRegistry {
    pub fn new() -> Self {
        Self { nodes: HashMap::new() }
    }

    /// Adiciona um novo nó à rede, após validação dos endossos.
    pub fn register_node(&mut self, record: NexusRecord) -> Result<()> {
        // Verificar se já existe
        if self.nodes.contains_key(&record.id) {
            return Err(ArkheError::ConstitutionViolation("Node already exists".into()));
        }

        // Verificar endossos (pelo menos 2 nós de épocas diferentes)
        if record.endorsements.len() < 2 {
            return Err(ArkheError::ConstitutionViolation("Insufficient endorsements".into()));
        }

        // Aqui poderia verificar se os endossantes são válidos...

        self.nodes.insert(record.id, record);
        Ok(())
    }

    pub fn get_node(&self, id: u64) -> Option<&NexusRecord> {
        self.nodes.get(&id)
    }

    pub fn list_active(&self) -> Vec<&NexusRecord> {
        self.nodes.values().filter(|n| n.active).collect()
    }
}
