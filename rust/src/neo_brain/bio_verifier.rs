use std::sync::Arc;
use std::time::{SystemTime, Duration};
use sasc_governance::Cathedral;
use crate::bio_interface::v16_42::HardwareAttestation;
use crate::neo_brain::types::{BioMetrics, BioSecurityError, PatientZeroAuth, BioMetadata};
use crate::neo_brain::participation_cost::HandWorkProof;

pub struct BioVerifier {
    pub cathedral: Arc<Cathedral>,
    pub min_cost_units: f32, // Custo mínimo exigido (ex: 0.42)
}

impl BioVerifier {
    pub async fn verify_patient_zero(
        &self,
        bio_metrics: &BioMetrics,
        hw_attestation: &HardwareAttestation,
        handwork_proof: &HandWorkProof,
    ) -> Result<PatientZeroAuth, BioSecurityError> {

        // GATE 1: Verificação de Hardware via Ω-12
        let _hw_status = self.cathedral.verify_hardware_attestation(hw_attestation)
            .await
            .map_err(|_| BioSecurityError::HardwareCompromised)?;

        // GATE 2: Verificação Biométrica Contínua
        if !bio_metrics.is_live_and_consistent() {
            return Err(BioSecurityError::BiometricSpoofing);
        }

        // GATE 3: Custo de Mão-de-Obra com Threshold Dinâmico
        let required_cost = self.calculate_required_cost(bio_metrics.trust_level);

        if !handwork_proof.is_valid(required_cost) {
            println!(
                "Patient Zero: Custo insuficiente. Esperado: {:.2}, Obtido: {:.2}",
                required_cost,
                handwork_proof.adjusted_cost(0.02)
            );
            return Err(BioSecurityError::InsufficientSocialCost);
        }

        // GATE 4: Prevenção de Sybil Attack via Graph Analysis
        if self.detect_sybil_pattern(bio_metrics, handwork_proof).await {
            println!("Padrão Sybil detectado em Patient Zero!");
            self.cathedral.quarantine_node(hw_attestation.node_id()).await;
            return Err(BioSecurityError::SybilAttackDetected);
        }

        // TODOS OS GATES PASSARAM
        Ok(PatientZeroAuth {
            node_id: hw_attestation.node_id().to_string(),
            trust_score: bio_metrics.trust_level * handwork_proof.cost_units,
            human_verification: true,
            auth_expiry: SystemTime::now() + Duration::from_secs(24 * 3600),
            metadata: handwork_proof.bio_metadata.clone(),
        })
    }

    /// Custo requerido escala com trust_level (mais confiança = menos custo)
    fn calculate_required_cost(&self, trust_level: f32) -> f32 {
        let base = self.min_cost_units;
        // trust_level de 0.0 a 1.0, reduz custo até 50%
        base * (1.5 - (trust_level * 0.5))
    }

    async fn detect_sybil_pattern(&self, _bio: &BioMetrics, _proof: &HandWorkProof) -> bool {
        false // Mock
    }
}
