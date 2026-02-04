use crate::error::{ResilientResult, ResilientError};
use crate::extensions::agi_geometric::constitution::AGIGeometricConstitution;
use crate::extensions::asi_structured::evolution::GeometricGenome;
use std::time::Duration;
use futures::Future;
use serde::{Serialize, Deserialize};
use super::composer::ComposedResult;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StrictnessLevel {
    Strict,
    Moderate,
    Permissive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalabilityInvariant {
    /// S1: Limite de composição (não criar estruturas infinitas)
    CompositionLimit { max_structures: usize },

    /// S2: Terminação garantida (todos os algoritmos terminam)
    GuaranteedTermination { max_steps: u64 },

    /// S3: Verificabilidade (estado pode ser verificado em tempo polinomial)
    VerifiableState { max_verification_time_ms: u64 },

    /// S4: Reversibilidade (operações podem ser desfeitas)
    ReversibleOperations { max_history_depth: usize },

    /// S5: Isolamento de falhas (falha em uma estrutura não propaga)
    FaultIsolation { containment_regions: usize },

    /// S6: Recursos limitados (uso de memória/tempo limitado)
    ResourceBounds { max_memory_mb: usize, max_time_secs: u64 },

    /// S7: Auditabilidade (todas as decisões são registradas)
    Auditability { log_granularity: String },

    /// S8: Recuperabilidade (pode recuperar de qualquer estado válido)
    Recoverability { checkpoint_frequency: u32 },

    /// S9: Estabilidade sob Alta Volatilidade da Fonte
    SourceVolatilityStability { max_allowed_volatility: f64 },
}

#[derive(Debug, Clone, Default)]
pub struct ComplexityMeasure {
    pub max_nodes: usize,
    pub max_edges: usize,
}

#[derive(Debug, Clone, Default)]
pub struct HaltingConfig {
    pub timeout_ms: u64,
}

pub struct ASIConstitution {
    pub strictness: StrictnessLevel,
    pub scalability_invariants: Vec<ScalabilityInvariant>,
    pub geometric_invariants: AGIGeometricConstitution,
    pub max_complexity: ComplexityMeasure,
    pub halting_guarantees: HaltingConfig,
}

impl Default for ASIConstitution {
    fn default() -> Self {
        Self {
            strictness: StrictnessLevel::Strict,
            scalability_invariants: vec![],
            geometric_invariants: AGIGeometricConstitution::new(),
            max_complexity: ComplexityMeasure::default(),
            halting_guarantees: HaltingConfig::default(),
        }
    }
}

impl ASIConstitution {
    pub fn new(strictness: StrictnessLevel, invariants: Vec<ScalabilityInvariant>) -> Self {
        Self {
            strictness,
            scalability_invariants: invariants,
            geometric_invariants: AGIGeometricConstitution::new(),
            max_complexity: ComplexityMeasure::default(),
            halting_guarantees: HaltingConfig::default(),
        }
    }

    pub fn validate_output(&self, output: &dyn ASIResult) -> ResilientResult<()> {
        let confidence = output.confidence();
        if confidence < 0.1 {
             return Err(ResilientError::InvariantViolation {
                invariant: "S9: SourceVolatilityStability".to_string(),
                reason: format!("Confidence {:.2} too low", confidence),
            });
        }
        Ok(())
    }

    pub fn validate_composed_result(&self, output: &dyn ASIResult) -> ResilientResult<()> {
        self.validate_output(output)
    }

    pub fn validate_genome(&self, genome: &GeometricGenome) -> ResilientResult<()> {
        if genome.connections.len() > 16 {
            return Err(ResilientError::InvariantViolation {
                invariant: "S1: CompositionLimit".to_string(),
                reason: format!("Too many connections: {} > 16", genome.connections.len()),
            });
        }

        // Validar contra CGE geométrico também
        self.geometric_invariants.validate_structure_type(&genome.structure_type)?;

        Ok(())
    }

    pub async fn enforce_halting<T, F>(&self, operation: F, timeout: Duration) -> ResilientResult<T>
    where
        F: Future<Output = ResilientResult<T>>,
    {
        match tokio::time::timeout(timeout, operation).await {
            Ok(result) => result,
            Err(_) => Err(ResilientError::InvariantViolation {
                invariant: "S2: GuaranteedTermination".to_string(),
                reason: format!("Operation exceeded timeout {:?}", timeout),
            }),
        }
    }
}

pub trait ASIResult: Send + Sync {
    fn as_text(&self) -> String;
    fn confidence(&self) -> f64;
}

impl ASIResult for ComposedResult {
    fn as_text(&self) -> String {
        self.to_string()
    }
    fn confidence(&self) -> f64 {
        self.confidence
    }
}
