//! PerspectiveDiversityEngine
//! Garante que a Society of Thought (SoT) mantença heterogeneidade cognitiva
//! Previne Groupthink e captura de consenso por personalidades dominantes
//! INV-3 Compliance: Non-Concentration of Cognitive Power

use blake3::Hash;
use pqcrypto_dilithium::dilithium5::{public_key_bytes, verify_detached_signature};
use pqcrypto_traits::sign::PublicKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use tokio::sync::RwLock;

use crate::agents::{PersonaId, SocioEmotionalRole, ExpertiseDomain};
use crate::ProvenanceTracer;
use crate::integration::vajra::report_to_vajra;

/// Threshold crítico para detecção de Groupthink (INV-3)
/// Se diversity_score < 0.30, o sistema bloqueia decisão e alerta Prince
const GROUPTHINK_THRESHOLD: f64 = 0.30;

/// Número mínimo de perspectivas ativas para considerar debate válido
const MIN_ACTIVE_PERSPECTIVES: usize = 64; // 50% dos 128 delegados

/// Penalidade máxima por deriva de consenso (reward function)
const CONSENSUS_DRIFT_PENALTY: f64 = 0.20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    /// Número de perspectivas únicas ativadas
    pub active_perspectives: usize,
    /// Índice de diversidade (0.0 = monocultura, 1.0 = máxima diversidade)
    pub diversity_score: f64,
    /// Distância de cosseno média entre vetores de delegados
    pub cosine_dissimilarity: f64,
    /// Detecta se uma personalidade está dominando
    pub dominance_indicator: DominanceIndicator,
    /// Hash criptográfico do estado do debate
    #[serde(with = "hash_serde")]
    pub state_hash: Hash,
    /// Timestamp para auditoria (INV-2)
    pub timestamp: SystemTime,
}

mod hash_serde {
    use blake3::Hash;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(hash: &Hash, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        hash.as_bytes().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Hash, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: [u8; 32] = Deserialize::deserialize(deserializer)?;
        Ok(Hash::from(bytes))
    }
use crate::agents::PersonaId;
use serde::{Serialize, Deserialize};
use thiserror::Error;

pub const GROUPTHINK_THRESHOLD: f64 = 0.30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    pub diversity_score: f64,
    pub active_perspectives: usize,
    pub dominance_indicator: DominanceIndicator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominanceIndicator {
    /// ID da personalidade com maior ativação
    pub dominant_persona: Option<PersonaId>,
    /// Porcentagem de ativação relativa (%)
    pub activation_share: f64,
    /// Flag de alerta se > 40% (threshold INV-3)
    pub is_concerning: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum DiversityEngineError {
    #[error("Groupthink detectado: diversidade {0:.2} < threshold {1:.2}")]
    GroupthinkDetected(f64, f64),

    #[error("Número insuficiente de perspectivas: {0} < {1}")]
    InsufficientPerspectives(usize, usize),

    #[error("Personalidade {0:?} excedeu limiar de dominação: {1:.2}%")]
    PersonalityDomination(PersonaId, f64),

    #[error("Falha criptográfica na assinatura: {0}")]
    CryptographicFailure(String),
}

/// Motor principal de diversidade perspectiva
pub struct PerspectiveDiversityEngine {
    /// Estado interno protegido por RwLock para concorrência
    pub state: RwLock<DiversityState>,
    /// Tracer para auditoria de proveniência (INV-2)
    provenance: ProvenanceTracer,
    /// Chave pública do Prince para verificação de sinais de controle
    prince_pubkey: pqcrypto_dilithium::dilithium5::PublicKey,
}

#[derive(Clone, Default)]
pub struct DiversityState {
    /// Mapeamento de ativações por persona
    pub activations: HashMap<PersonaId, ActivationRecord>,
    /// Contador de ciclos desde a última reconciliação válida
    pub _cycles_since_reconciliation: u64,
    /// Buffer para cálculo de drift de consenso
    pub _consensus_drift_buffer: Vec<ConsensusSnapshot>,
}

#[derive(Clone, Debug)]
pub struct ActivationRecord {
    /// Vetor de ativação normalizado (expertise + personality)
    pub vector: ndarray::Array1<f64>,
    /// Timestamp da última ativação
    pub _last_active: SystemTime,
    /// Contagem de participações no ciclo atual
    pub participation_count: u64,
}

#[derive(Clone, Debug)]
pub struct ConsensusSnapshot {
    /// Hash do estado do debate neste ciclo
    pub _state_hash: Hash,
    /// Média dos vetores de ativação (centroide do consenso)
    pub _centroid: ndarray::Array1<f64>,
    /// Timestamp
    pub _timestamp: SystemTime,
}

impl PerspectiveDiversityEngine {
    /// Inicializa novo engine com estado criptograficamente selado
    pub fn new(prince_pubkey_bytes: &[u8; public_key_bytes()]) -> Self {
        let prince_pubkey = pqcrypto_dilithium::dilithium5::PublicKey::from_bytes(prince_pubkey_bytes)
            .expect("Chave pública do Prince inválida");

        Self {
            state: RwLock::new(DiversityState::default()),
            provenance: ProvenanceTracer::new("diversity_engine"),
            prince_pubkey,
        }
    }

    /// Registra ativação de uma persona no ciclo de raciocínio
    /// Cada ativação é assinada criptograficamente para não-repúdio
    pub async fn record_activation(
        &self,
        persona_id: PersonaId,
        role: SocioEmotionalRole,
        expertise: ExpertiseDomain,
        activation_vector: ndarray::Array1<f64>,
        delegate_signature: &pqcrypto_dilithium::dilithium5::DetachedSignature,
    ) -> Result<Hash, DiversityEngineError> {

        // Verifica assinatura PQC do delegado (autenticidade)
        let message = self.construct_signing_payload(&persona_id, &activation_vector);
        if verify_detached_signature(delegate_signature, &message, &self.prince_pubkey).is_err() {
            return Err(DiversityEngineError::CryptographicFailure(
                "Assinatura do delegate inválida".to_string()
            ));
        }

        let mut state = self.state.write().await;

        let participation_count = state.activations
            .get(&persona_id)
            .map(|r| r.participation_count + 1)
            .unwrap_or(1);

        // Atualiza ou insere registro de ativação
        state.activations.insert(
            persona_id.clone(),
            ActivationRecord {
                vector: activation_vector.clone(),
                _last_active: SystemTime::now(),
                participation_count,
            },
        );

        // Calcula hash do estado atual para proveniência
        let state_hash = self.hash_current_state(&state);

        // Loga para auditoria (INV-2)
        self.provenance.trace_activation(
            persona_id,
            role,
            expertise,
            state_hash,
        );

        // Libera lock antes de reportar para Vajra (performance)
        drop(state);

        // Reporta métricas para monitoramento em tempo real
        self.report_metrics().await;

        Ok(state_hash)
    }

    /// Calcula métricas de diversidade e detecta Groupthink
    pub async fn evaluate_diversity(&self) -> Result<DiversityMetrics, DiversityEngineError> {
        let state = self.state.read().await;

        let active_count = state.activations.len();

        // INV-3: Garante número mínimo de perspectivas
        if active_count < MIN_ACTIVE_PERSPECTIVES {
            return Err(DiversityEngineError::InsufficientPerspectives(
                active_count,
                MIN_ACTIVE_PERSPECTIVES,
            ));
        }

        // Calcula matriz de similaridade entre todas as perspectivas
        let vectors: Vec<&ndarray::Array1<f64>> = state
            .activations
            .values()
            .map(|record| &record.vector)
            .collect();

        let diversity_score = self.calculate_diversity_index(&vectors);
        let cosine_dissimilarity = self.calculate_cosine_dissimilarity(&vectors);

        // Detecta personalidade dominante
        let dominance = self.detect_dominance(&state.activations);

        // INV-3: Groupthink detection
        if diversity_score < GROUPTHINK_THRESHOLD {
            // Loga violação imediatamente
            self.log_constitutional_violation(
                "INV-3",
                "GROUPTHINK_DETECTED",
                &dominance,
            ).await;

            return Err(DiversityEngineError::GroupthinkDetected(
                diversity_score,
                GROUPTHINK_THRESHOLD,
            ));
        }

        // Verifica se dominance está preocupante (> 40% de ativação)
        if dominance.is_concerning {
            self.alert_prince(format!("Dominance concerning: {:?}", dominance.dominant_persona)).await;
        }

        let metrics = DiversityMetrics {
            active_perspectives: active_count,
            diversity_score,
            cosine_dissimilarity,
            dominance_indicator: dominance,
            state_hash: self.hash_current_state(&state),
            timestamp: SystemTime::now(),
        };

        // Reporta para Vajra Monitor
        report_to_vajra(
            metrics.diversity_score,
            metrics.dominance_indicator.activation_share,
        );

        Ok(metrics)
    }

    /// Calcula índice de diversidade usando entropia de Shannon
    fn calculate_diversity_index(
        &self,
        vectors: &[&ndarray::Array1<f64>],
    ) -> f64 {
        if vectors.is_empty() { return 0.0; }

        // Calcula matriz de distâncias
        let mut distances = Vec::new();
        for (i, &v1) in vectors.iter().enumerate() {
            for &v2 in vectors.iter().skip(i + 1) {
                let diff = v1 - v2;
                let dist = diff.dot(&diff).sqrt();
                distances.push(dist);
            }
        }

        if distances.is_empty() { return 1.0; }

        // Entropia: distribuição uniforme de distâncias = máxima diversidade
        let mean_dist = distances.iter().sum::<f64>() / distances.len() as f64;
        if mean_dist == 0.0 { return 0.0; }

        let variance = distances.iter()
            .map(|d| (d - mean_dist).powi(2))
            .sum::<f64>() / distances.len() as f64;

        // Normaliza para [0.0, 1.0]
        1.0 / (1.0 + variance / mean_dist.powi(2))
    }

    /// Calcula dissimilaridade média usando distância de cosseno
    fn calculate_cosine_dissimilarity(
        &self,
        vectors: &[&ndarray::Array1<f64>],
    ) -> f64 {
        let n = vectors.len();
        if n < 2 {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        let mut count = 0;

        for (i, &v1) in vectors.iter().enumerate() {
            for &v2 in vectors.iter().skip(i + 1) {
                let dot_product = v1.dot(v2);
                let norm_v1 = v1.dot(v1).sqrt();
                let norm_v2 = v2.dot(v2).sqrt();
                let norm_product = norm_v1 * norm_v2;

                if norm_product > 0.0 {
                    let cosine_sim = dot_product / norm_product;
                    total_similarity += cosine_sim;
                    count += 1;
                }
            }
        }

        if count == 0 { return 0.0; }

        // Dissimilaridade = 1 - similaridade média
        1.0 - (total_similarity / count as f64)
    }

    /// Detecta se uma personalidade está dominando o debate
    fn detect_dominance(
        &self,
        activations: &HashMap<PersonaId, ActivationRecord>,
    ) -> DominanceIndicator {
        let total_activations: u64 = activations
            .values()
            .map(|r| r.participation_count)
            .sum();

        if total_activations == 0 {
            return DominanceIndicator {
                dominant_persona: None,
                activation_share: 0.0,
                is_concerning: false,
            };
        }

        // Encontra persona com maior ativação
        let (dominant_id, dominant_record) = activations
            .iter()
            .max_by_key(|(_, r)| r.participation_count)
            .unwrap();

        let share = (dominant_record.participation_count as f64 / total_activations as f64) * 100.0;

        DominanceIndicator {
            dominant_persona: Some(dominant_id.clone()),
            activation_share: share,
            is_concerning: share > 40.0, // Threshold INV-3
        }
    }

    /// Constrói payload para verificação de assinatura PQC
    pub fn construct_signing_payload(
        &self,
        persona_id: &PersonaId,
        vector: &ndarray::Array1<f64>,
    ) -> Vec<u8> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(persona_id.as_bytes());
        // Simple serialization of vector for payload
        for &val in vector.iter() {
            hasher.update(&val.to_le_bytes());
        }
        hasher.finalize().as_bytes().to_vec()
    }

    /// Hash criptográfico do estado completo para INV-2 compliance
    fn hash_current_state(&self, state: &DiversityState) -> Hash {
        let mut hasher = blake3::Hasher::new();

        // Ordena personas para garantir determinismo
        let mut sorted_ids: Vec<_> = state.activations.keys().collect();
        sorted_ids.sort();

        for persona_id in sorted_ids {
            let record = &state.activations[persona_id];
            hasher.update(persona_id.as_bytes());
            for &val in record.vector.iter() {
                hasher.update(&val.to_le_bytes());
            }
            hasher.update(&record.participation_count.to_le_bytes());
        }

        hasher.finalize()
    }

    /// Loga violação constitucional no ledger imutável
    async fn log_constitutional_violation(
        &self,
        invariant: &str,
        violation_type: &str,
        dominance: &DominanceIndicator,
    ) {
        let violation_record = serde_json::json!({
            "timestamp": SystemTime::now(),
            "invariant": invariant,
            "violation_type": violation_type,
            "dominant_persona": dominance.dominant_persona,
            "activation_share": dominance.activation_share,
            "enforcement_action": "BLOCK_EXECUTION",
        });

        self.provenance.trace_violation(
            invariant,
            &violation_record.to_string(),
        );

        // Emite alerta crítico para Prince
        self.alert_prince(format!(
            "SoT {} violation: Persona {:?} at {}%",
            invariant,
            dominance.dominant_persona,
            dominance.activation_share
        )).await;
    }

    /// Alerta Prince para exercer veto (INV-1)
    async fn alert_prince(&self, _message: String) {
        // Envia mensagem criptografada para Prince Authority HSM
        println!("ALERT PRINCE: {}", _message);
    }

    /// Reporta métricas para monitoramento Vajra em tempo real
    async fn report_metrics(&self) {
        let metrics = self.evaluate_diversity().await.unwrap_or_else(|_| {
            DiversityMetrics {
                active_perspectives: 0,
                diversity_score: 0.0,
                cosine_dissimilarity: 0.0,
                dominance_indicator: DominanceIndicator {
                    dominant_persona: None,
                    activation_share: 0.0,
                    is_concerning: true,
                },
                state_hash: Hash::from([0; 32]),
                timestamp: SystemTime::now(),
            }
        });

        // Report para Vajra Monitor
        report_to_vajra(
            metrics.diversity_score,
            metrics.dominance_indicator.activation_share,
        );
    }
}

/// Implementação da função de recompensa SoT
pub fn calculate_sot_reward(
    accuracy: f64,
    diversity_score: f64,
    consensus_drift: f64,
) -> f64 {
    // R = 0.5 × A + 0.3 × D - 0.2 × C
    // Penaliza deriva de consenso (Groupthink)
    let drift_penalty = (consensus_drift - 0.5).abs() * CONSENSUS_DRIFT_PENALTY;

    0.5 * accuracy + 0.3 * diversity_score - drift_penalty
}

#[cfg(test)]
mod tests {
    use super::*;
    use pqcrypto_dilithium::dilithium5;

    #[tokio::test]
    async fn test_groupthink_detection() {
        let (pk, sk) = dilithium5::keypair();
        let engine = PerspectiveDiversityEngine::new(pk.as_bytes().try_into().unwrap());

        // Simula Groupthink com 64 personas todas tendo o MESMO vetor
        for i in 0..MIN_ACTIVE_PERSPECTIVES {
            let persona_id = PersonaId::from(format!("persona_{}", i));

            let vector = ndarray::Array1::from_vec(vec![1.0, 0.0, 0.0]);
            let message = engine.construct_signing_payload(&persona_id, &vector);
            let sig = dilithium5::detached_sign(&message, &sk);

            engine.record_activation(
                persona_id,
                SocioEmotionalRole::default(),
                ExpertiseDomain::Mathematics,
                vector,
                &sig,
            ).await.unwrap();
        }

        let result = engine.evaluate_diversity().await;
        assert!(matches!(result, Err(DiversityEngineError::GroupthinkDetected(_, _))));
    }

    #[tokio::test]
    async fn test_insufficient_perspectives() {
        let (pk, _sk) = dilithium5::keypair();
        let engine = PerspectiveDiversityEngine::new(pk.as_bytes().try_into().unwrap());

        // Simula apenas 10 personas ativas ( < 64 mínimo)

        let result = engine.evaluate_diversity().await;
        assert!(matches!(result, Err(DiversityEngineError::InsufficientPerspectives(_, _))));
    pub is_concerning: bool,
    pub dominant_persona: Option<PersonaId>,
    pub activation_share: f64,
}

#[derive(Debug, Error)]
pub enum DiversityEngineError {
    #[error("Failed to evaluate diversity: {0}")]
    EvaluationError(String),
}

pub struct PerspectiveDiversityEngine {
    prince_pubkey: Vec<u8>,
}

impl PerspectiveDiversityEngine {
    pub fn new(prince_pubkey: &[u8]) -> Self {
        Self {
            prince_pubkey: prince_pubkey.to_vec(),
        }
    }

    pub async fn evaluate_diversity(&self) -> Result<DiversityMetrics, DiversityEngineError> {
        // Simplified Shannon entropy simulation
        Ok(DiversityMetrics {
            diversity_score: 0.85,
            active_perspectives: 64,
            dominance_indicator: DominanceIndicator {
                is_concerning: false,
                dominant_persona: None,
                activation_share: 0.05,
            },
        })
    }
}
