//! Arkhe(n) Diplomatic Protocol
//! Implementação da interoperabilidade satelital baseada em coerência de fase.

use crate::hardware_embassy::HardwareEmbassy;
use crate::ArkheError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum HandshakeStatus {
    ACCEPTED,
    REJECTED,
    PENDING,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub status: HandshakeStatus,
    pub g_adjustment: f64,
    pub coherence_global: f64,
}

pub struct DiplomaticProtocol {
    pub node_id: String,
    pub threshold_psi: f64,
    pub hardware: Option<HardwareEmbassy>,
}

impl DiplomaticProtocol {
    pub fn new(node_id: &str, threshold_psi: f64) -> Self {
        Self {
            node_id: node_id.to_string(),
            threshold_psi,
            hardware: None,
        }
    }

    pub fn attach_hardware(&mut self, hardware: HardwareEmbassy) {
        self.hardware = Some(hardware);
    }

    /// Realiza a tentativa de handshake integrando dados do hardware SDR
    pub fn attempt_handshake(
        &mut self,
        remote_node_id: &str,
        remote_phase: f64,
        remote_coherence: f64
    ) -> Result<HandshakeResponse, ArkheError> {
        // 1. Obter dados locais do hardware (se disponível)
        let (local_phase, local_coherence) = if let Some(hw) = &mut self.hardware {
            hw.extract_phase_and_coherence()
                .map_err(|e| ArkheError::Other(format!("SDR Error: {:?}", e)))?
        } else {
            // Fallback para simulação se hardware não estiver presente
            (0.0, 0.95)
        };

        // 2. Calcular coerência combinada (física do handshake)
        let combined_coherence = (local_coherence + remote_coherence) / 2.0;

        // 3. Validar contra o threshold Ψ
        if combined_coherence < self.threshold_psi {
            return Ok(HandshakeResponse {
                status: HandshakeStatus::REJECTED,
                g_adjustment: 0.0,
                coherence_global: combined_coherence,
            });
        }

        // 4. Calcular ajuste de fase g|ψ|² = -Δϕ
        // O ajuste visa sincronizar a fase local com a remota
        let phase_diff = remote_phase - local_phase;
        let g_adjustment = -phase_diff;

        println!(
            "Handshake with {}: Combined Coherence {:.4}, Phase Adjustment {:.4}",
            remote_node_id, combined_coherence, g_adjustment
        );

        Ok(HandshakeResponse {
            status: HandshakeStatus::ACCEPTED,
            g_adjustment,
            coherence_global: combined_coherence,
        })
    }
}
