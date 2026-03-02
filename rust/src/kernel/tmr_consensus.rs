// kernel/tmr_consensus.rs
// Consenso por maioria entre 3 kernels para decisões críticas

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use tokio::sync::{RwLock, broadcast};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TMRConsensusError {
    #[error("Insufficient kernels for consensus")]
    InsufficientKernels,
    #[error("Phi divergence too high: {0}")]
    PhiDivergence(f64),
    #[error("Constitutional divergence too high")]
    ConstitutionalDivergence,
    #[error("Prince signature required")]
    PrinceSignatureRequired,
    #[error("Phi too low: {0}")]
    PhiTooLow(f64),
    #[error("Insufficient approval: {0}/{1}")]
    InsufficientApproval(usize, usize),
    #[error("Signature error: {0}")]
    SignatureError(#[from] ed25519_dalek::SignatureError),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ConsensusResult {
    pub global_phi: f64,
    pub lyapunov_sigma: f64,
    pub constitutional_alignment: f64,
    pub kernel_agreement: f64,
    pub timestamp: u128,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CriticalDecision {
    pub action: String,
    pub details: String,
}

pub struct DecisionResult {
    pub executed: bool,
    pub consensus: ConsensusResult,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct KernelHeartbeat {
    pub kernel_id: String,
    pub timestamp: u128,
    pub phi_measurement: f64,
    pub lyapunov_sigma: f64,
    pub constitutional_hash: [u8; 32],
    pub signature: Vec<u8>,
}

impl KernelHeartbeat {
    pub fn signing_data(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(self.kernel_id.as_bytes());
        data.extend_from_slice(&self.timestamp.to_le_bytes());
        data.extend_from_slice(&self.phi_measurement.to_le_bytes());
        data.extend_from_slice(&self.lyapunov_sigma.to_le_bytes());
        data.extend_from_slice(&self.constitutional_hash);
        data
    }
}

pub struct TMRConsensus {
    kernels: RwLock<HashMap<String, KernelHeartbeat>>,
    prince_key: VerifyingKey,
    shadower_keys: Vec<VerifyingKey>,
    consensus_channel: broadcast::Sender<ConsensusResult>,
}

impl TMRConsensus {
    pub async fn new(prince_key: VerifyingKey) -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
            prince_key,
            shadower_keys: Vec::new(),
            consensus_channel: broadcast::channel(100).0,
        }
    }

    pub async fn submit_heartbeat(&self, heartbeat: KernelHeartbeat) -> Result<(), TMRConsensusError> {
        // 1. Verificar assinatura do kernel
        let kernel_key = self.get_kernel_key(&heartbeat.kernel_id).await?;
        let sig = Signature::from_slice(&heartbeat.signature)?;
        kernel_key.verify(&heartbeat.signing_data(), &sig)?;

        // 2. Armazenar heartbeat
        self.kernels.write().await.insert(heartbeat.kernel_id.clone(), heartbeat);

        // 3. Verificar consenso a cada 3 heartbeats
        if self.kernels.read().await.len() >= 3 {
            self.check_consensus().await?;
        }

        Ok(())
    }

    pub async fn check_consensus(&self) -> Result<ConsensusResult, TMRConsensusError> {
        let kernels = self.kernels.read().await;

        if kernels.len() < 3 {
            return Err(TMRConsensusError::InsufficientKernels);
        }

        // Coletar métricas de todos os kernels
        let mut phi_measurements = Vec::new();
        let mut sigma_measurements = Vec::new();
        let mut constitutional_hashes = Vec::new();

        for (_, heartbeat) in kernels.iter() {
            phi_measurements.push(heartbeat.phi_measurement);
            sigma_measurements.push(heartbeat.lyapunov_sigma);
            constitutional_hashes.push(heartbeat.constitutional_hash);
        }

        // Verificar divergência de Φ (threshold: 0.01)
        let phi_mean = mean(&phi_measurements);
        let phi_std = std_dev(&phi_measurements);

        if phi_std > 0.01 {
            // Discrepância crítica - disparar KARNAK level3
            self.trigger_karnak(
                "level3",
                format!("Divergência de Φ: σ={:.4}", phi_std)
            ).await?;

            return Err(TMRConsensusError::PhiDivergence(phi_std));
        }

        // Verificar consenso constitucional
        let constitutional_consensus = self.check_constitutional_consensus(&constitutional_hashes);
        if constitutional_consensus < 0.67 {
            return Err(TMRConsensusError::ConstitutionalDivergence);
        }

        // Calcular Φ global (média dos 3 kernels)
        let global_phi = phi_mean;

        // Verificar estabilidade de Lyapunov
        let max_sigma = sigma_measurements.iter().cloned().fold(0.0, f64::max);
        if max_sigma > 0.00005 {
            self.trigger_karnak(
                "level4",
                format!("Instabilidade Lyapunov: σ={:.6}", max_sigma)
            ).await?;
        }

        // Emitir resultado de consenso
        let result = ConsensusResult {
            global_phi,
            lyapunov_sigma: max_sigma,
            constitutional_alignment: constitutional_consensus,
            kernel_agreement: 1.0, // 3.0 / 3.0
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        };

        // Broadcast para todos os componentes
        let _ = self.consensus_channel.send(result.clone());

        Ok(result)
    }

    pub async fn critical_decision(&self, decision: CriticalDecision) -> Result<DecisionResult, TMRConsensusError> {
        // Decisões críticas requerem:
        // 1. Assinatura Prince
        // 2. Consenso 2/3 dos kernels
        // 3. Φ > 0.78

        // Verificar assinatura Prince
        if !self.verify_prince_signature(&decision).await {
            return Err(TMRConsensusError::PrinceSignatureRequired);
        }

        // Verificar consenso atual
        let consensus = self.check_consensus().await?;

        if consensus.global_phi < 0.78 {
            return Err(TMRConsensusError::PhiTooLow(consensus.global_phi));
        }

        // Coletar votos dos kernels
        let votes = self.collect_kernel_votes(&decision).await;

        // Requer 2/3 aprovação
        let approval_count = votes.iter().filter(|&v| *v).count();
        if approval_count < 2 {
            return Err(TMRConsensusError::InsufficientApproval(approval_count, 3));
        }

        // Executar decisão
        let result = self.execute_decision(decision, consensus).await;

        Ok(result)
    }

    async fn get_kernel_key(&self, _kernel_id: &str) -> Result<VerifyingKey, TMRConsensusError> {
        // Mock returning the prince key as a fallback kernel key for simulation
        Ok(self.prince_key)
    }

    async fn trigger_karnak(&self, level: &str, reason: String) -> Result<(), TMRConsensusError> {
        println!("[KARNAK-TMR] Triggering {} due to: {}", level, reason);
        Ok(())
    }

    fn check_constitutional_consensus(&self, hashes: &[[u8; 32]]) -> f64 {
        if hashes.is_empty() { return 0.0; }
        // For simulation, assume all agree
        1.0
    }

    async fn verify_prince_signature(&self, _decision: &CriticalDecision) -> bool {
        // For simulation, assume valid
        true
    }

    async fn collect_kernel_votes(&self, _decision: &CriticalDecision) -> Vec<bool> {
        // For simulation, 3 votes for yes
        vec![true, true, true]
    }

    async fn execute_decision(&self, _decision: CriticalDecision, consensus: ConsensusResult) -> DecisionResult {
        DecisionResult {
            executed: true,
            consensus,
        }
    }
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let m = mean(data);
    let variance = data.iter().map(|value| {
        let diff = m - (*value);
        diff * diff
    }).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}
