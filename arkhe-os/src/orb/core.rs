// arkhe-os/src/orb/core.rs

use serde::{Deserialize, Serialize};
use super::Error;

/// A estrutura mínima de um Orb — independente de protocolo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbPayload {
    /// Identificador único do Orb
    pub orb_id: [u8; 32],

    /// Coerência acumulada
    pub lambda_2: f64,

    /// Fase quântica
    pub phi_q: f64,

    /// Restrição termodinâmica
    pub h_value: f64,

    /// Origem temporal (ex: 2026)
    pub origin_time: i64,

    /// Destino temporal (ex: 2008)
    pub target_time: i64,

    /// Hash da Timechain
    pub timechain_hash: [u8; 32],

    /// Assinatura PQC (Dilithium3)
    pub signature: Vec<u8>,

    /// Timestamp de criação
    pub created_at: i64,

    /// Delta de estado evolutivo (Cortex/Whittaker context)
    pub state_delta: Option<StateDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateDelta {
    pub new_context: Vec<u8>,
    pub memory_update: Vec<u8>,
    pub decay_rate: f64,
}

impl OrbPayload {
    /// Serializa para bytes (formato canônico)
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    /// Desserializa de bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, Error> {
        bincode::deserialize(data).map_err(|e| Error::Deserialization(e.to_string()))
    }

    /// Calcula o "peso" informacional do Orb
    pub fn informational_mass(&self) -> f64 {
        self.lambda_2 * self.phi_q / self.h_value.max(0.001)
    }

    pub fn is_retrocausal(&self) -> bool {
        self.target_time > self.origin_time
    }
}
