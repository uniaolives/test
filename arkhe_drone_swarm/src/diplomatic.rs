//! Arkhe(n) Diplomatic Protocol
//! Implementação da interoperabilidade satelital baseada em coerência de fase.

use crate::hardware_embassy::HardwareEmbassy;
use crate::kalman::AdaptiveKalmanPredictor;
use crate::zk_lattice::{ArkheZKProver, Polynomial};
use crate::ArkheError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Copy)]
pub enum HandshakeStatus {
    ACCEPTED,
    REJECTED,
    PENDING,
    SemionicFallback,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Copy)]
pub enum ProtocolState {
    Normal,
    Semionic,  // alpha = 0.5
    Annealing, // Transição de volta para Normal
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub status: HandshakeStatus,
    pub g_adjustment: f64,
    pub coherence_global: f64,
    pub alpha: f64,
    #[serde(skip)]
    pub zk_proof: Option<Polynomial>,
}

pub struct DiplomaticProtocol {
    pub node_id: String,
    pub threshold_psi: f64,
    pub hardware: Option<HardwareEmbassy>,
    pub state: ProtocolState,
    pub current_alpha: f64,
    pub target_alpha: f64,
    pub kalman: AdaptiveKalmanPredictor,
    pub last_timestamp: u64, // ms
    pub prover: Option<ArkheZKProver>,
}

impl DiplomaticProtocol {
    pub fn new(node_id: &str, threshold_psi: f64) -> Self {
        let golden_ratio = 0.61803398875;
        Self {
            node_id: node_id.to_string(),
            threshold_psi,
            hardware: None,
            state: ProtocolState::Normal,
            current_alpha: golden_ratio,
            target_alpha: golden_ratio,
            kalman: AdaptiveKalmanPredictor::new(1e-4, 1e-2, 30.0),
            last_timestamp: 0,
            prover: None,
        }
    }

    pub fn set_prover(&mut self, prover: ArkheZKProver) {
        self.prover = Some(prover);
    }

    pub fn attach_hardware(&mut self, hardware: HardwareEmbassy) {
        self.hardware = Some(hardware);
    }

    /// Realiza a tentativa de handshake integrando dados do hardware SDR
    pub fn attempt_handshake(
        &mut self,
        remote_node_id: &str,
        remote_phase: f64,
        remote_coherence: f64,
        timestamp: u64,
    ) -> Result<HandshakeResponse, ArkheError> {
        let dt = if self.last_timestamp == 0 {
            0.01 // 10ms default
        } else {
            (timestamp - self.last_timestamp) as f64 / 1000.0
        };
        self.last_timestamp = timestamp;

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

        // 2.1. Atualizar Filtro de Kalman Adaptativo
        self.kalman.update(remote_phase, dt, combined_coherence);
        let predicted_phase = self.kalman.predict_phase(dt);

        // 3. Gerenciar estados de Fallback e Annealing
        let mut status = HandshakeStatus::ACCEPTED;

        if combined_coherence < self.threshold_psi {
            if self.state != ProtocolState::Semionic {
                println!("⚠️ [SAFE] Coerência crítica ({:.3}). Iniciando Fallback Semiónico.", combined_coherence);
                self.state = ProtocolState::Semionic;
                self.current_alpha = 0.5;
                status = HandshakeStatus::SemionicFallback;
            } else {
                return Ok(HandshakeResponse {
                    status: HandshakeStatus::REJECTED,
                    g_adjustment: 0.0,
                    coherence_global: combined_coherence,
                    alpha: self.current_alpha,
                    zk_proof: None,
                });
            }
        } else if self.state == ProtocolState::Semionic {
            println!("💡 [SAFE] Coerência restaurada ({:.3}). Iniciando Annealing.", combined_coherence);
            self.state = ProtocolState::Annealing;
        }

        // 4. Executar Annealing se necessário
        if self.state == ProtocolState::Annealing {
            self.step_annealing();
        }

        // 5. Calcular ajuste de fase g|ψ|² = -Δϕ
        // Usamos a fase predita pelo Kalman se a coerência for baixa
        let effective_remote_phase = if combined_coherence < 0.5 {
            predicted_phase
        } else {
            remote_phase
        };

        let phase_diff = effective_remote_phase - local_phase;
        let g_adjustment = -phase_diff;

        println!(
            "Handshake with {}: State {:?}, Coherence {:.4}, Alpha {:.4}",
            remote_node_id, self.state, combined_coherence, self.current_alpha
        );

        // 6. Gerar Prova ZK da fase física (Post-Quantum)
        let zk_proof = self.prover.as_ref().map(|p| p.generate_phase_proof(local_phase));

        Ok(HandshakeResponse {
            status,
            g_adjustment,
            coherence_global: combined_coherence,
            alpha: self.current_alpha,
            zk_proof,
        })
    }

    /// Incrementa o processo de annealing para retornar à fase áurea
    pub fn step_annealing(&mut self) {
        let step = 0.01;
        if self.current_alpha < self.target_alpha {
            self.current_alpha = (self.current_alpha + step).min(self.target_alpha);
        } else if self.current_alpha > self.target_alpha {
            self.current_alpha = (self.current_alpha - step).max(self.target_alpha);
        }

        if (self.current_alpha - self.target_alpha).abs() < 1e-6 {
            println!("✅ [SAFE] Annealing completo. Retornado ao estado Normal.");
            self.state = ProtocolState::Normal;
        }
    }
}
