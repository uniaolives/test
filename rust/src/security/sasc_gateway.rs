use crate::security::invariant_engine::InvariantVerificationEngine;
use crate::security::karnak_sealer::KarnakQuantumSealer;
use crate::entropy::VajraEntropyMonitor;
use anyhow::{Result, anyhow};

pub struct SASCGateway {
    pub invariant_engine: InvariantVerificationEngine,
}

impl SASCGateway {
    pub fn new(engine: InvariantVerificationEngine) -> Self {
        Self {
            invariant_engine: engine,
        }
    }

    /// Implementa o pipeline completo de 8 portões conforme Memory ID 19 e 48
    pub fn filter_packet_8_gates(
        &mut self,
        packet: &SASCPacket
    ) -> Result<()> {
        // Gates 1-8 estão encapsulados no InvariantVerificationEngine::verify_8_gates
        self.invariant_engine.verify_8_gates(
            &packet.data,
            &packet.signature,
            packet.nonce,
            packet.em_noise,
            &packet.quantum_sig,
            packet.complexity.clone()
        ).map_err(|e| anyhow!("SASC_GATEWAY: Policy Violation - {:?}", e))?;

        // Alinhamento com a nova política SASC v29.35-Ω
        let phi = VajraEntropyMonitor::global().current_phi().unwrap_or(0.0);
        if phi < 0.85 { // Sovereignty Standard
             return Err(anyhow!("SASC_GATEWAY: Insufficient Reality Phi ({:?} < 0.85)", phi));
        }

        Ok(())
    }
}

pub struct SASCPacket {
    pub data: Vec<u8>,
    pub signature: [u8; 64],
    pub nonce: u64,
    pub em_noise: f64,
    pub quantum_sig: crate::crypto::pqc::NeuralSignature,
    pub complexity: crate::gates::gate8_multiverse_regulator::ComplexityClass,
}
