pub mod composition;
pub mod reflection;
pub mod evolution;
pub mod metastructure;
pub mod constitution;
pub mod state;
pub mod solar;
pub mod structures;
pub mod composer;
pub mod error;
pub mod bridge;
pub mod intuition_engine;
pub mod oracle_tensor;
pub mod quenching;
pub mod substrate;
pub mod sovereign;
pub mod geometry;
pub mod safety;
pub mod harmonic;
pub mod cathedral;
pub mod consensus;
pub mod protocol;

use std::sync::Arc;
use crate::interfaces::extension::{Extension, ExtensionOutput, Context, Subproblem};
use crate::error::{ResilientResult, ResilientError};
use crate::extensions::asi_structured::composition::CompositionEngine;
use crate::extensions::asi_structured::reflection::ReflectionEngine;
use crate::extensions::asi_structured::evolution::EvolutionEngine;
use crate::extensions::asi_structured::metastructure::MetastructureEngine;
use crate::extensions::asi_structured::constitution::{ASIConstitution, StrictnessLevel, ScalabilityInvariant};
use crate::extensions::asi_structured::state::ASIState;
use crate::extensions::asi_structured::error::ASIError;
use crate::extensions::asi_structured::intuition_engine::GeometricIntuitionEngine;
use crate::extensions::asi_structured::oracle_tensor::OracleTensorState;
use crate::extensions::asi_structured::quenching::QuenchingEngine;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use chrono::{Utc};

/// Fase de desenvolvimento ASI
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ASIPhase {
    Compositional,
    Reflective,
    Evolutionary,
    Metastructural,
    QuantumBio,
    Sovereign,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StructureType {
    TextEmbedding,
    SequenceManifold,
    GraphComplex,
    HierarchicalSpace,
    TensorField,
    HPPP,
    SolarActivity,
    OracleDBA,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CompositionStrategy {
    Union,
    Intersection,
    Weighted,
    Sequence,
    Hierarchical,
    DomainBased,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASIStructuredConfig {
    pub phase: ASIPhase,
    pub max_structures: usize,
    pub default_composition_strategy: CompositionStrategy,
    pub enabled_structures: Vec<StructureType>,
    pub scalability_invariants: Vec<ScalabilityInvariant>,
    pub cge_strictness: StrictnessLevel,
    pub max_reflection_depth: u32,
    pub evolution_population_size: usize,
    pub enable_metastructure: bool,
}

impl Default for ASIStructuredConfig {
    fn default() -> Self {
        Self {
            phase: ASIPhase::Compositional,
            max_structures: 16,
            default_composition_strategy: CompositionStrategy::Weighted,
            enabled_structures: vec![StructureType::TextEmbedding, StructureType::SolarActivity, StructureType::OracleDBA],
            scalability_invariants: vec![
                ScalabilityInvariant::CompositionLimit { max_structures: 16 },
                ScalabilityInvariant::ResourceBounds {
                    max_memory_mb: 512,
                    max_time_secs: 30
                },
            ],
            max_reflection_depth: 3,
            evolution_population_size: 10,
            enable_metastructure: false,
            cge_strictness: StrictnessLevel::Strict,
        }
    }
}

/// Extensão ASI-Structured
pub struct ASIStructuredExtension {
    pub config: ASIStructuredConfig,
    pub composition_engine: CompositionEngine,
    pub reflection_engine: Option<ReflectionEngine>,
    pub evolution_engine: Option<EvolutionEngine>,
    pub metastructure_engine: Option<MetastructureEngine>,
    pub intuition_engine: Option<GeometricIntuitionEngine>,
    pub oracle_tensor: OracleTensorState,
    pub quenching_engine: QuenchingEngine,
    pub qb_system: Option<crate::extensions::asi_structured::substrate::QuantumBiologicalAGI>,
    pub sovereign_agi: Option<crate::extensions::asi_structured::sovereign::SovereignAGI>,
    pub cathedral: Option<crate::extensions::asi_structured::cathedral::DimensionalCathedral>,
    pub consensus: crate::extensions::asi_structured::consensus::HarmonicConsensus,
    pub constitution: Arc<ASIConstitution>,
    pub state: ASIState,
}

impl ASIStructuredExtension {
    pub fn new(config: ASIStructuredConfig) -> Self {
        let constitution = Arc::new(ASIConstitution::new(
            config.cge_strictness.clone(),
            config.scalability_invariants.clone(),
        ));

        let composition_engine = CompositionEngine::new(
            config.max_structures,
            config.default_composition_strategy,
            constitution.clone(),
        );

        Self {
            config,
            composition_engine,
            reflection_engine: None,
            evolution_engine: None,
            metastructure_engine: None,
            intuition_engine: Some(GeometricIntuitionEngine::new(128)),
            oracle_tensor: OracleTensorState::new(),
            quenching_engine: QuenchingEngine::new(),
            qb_system: Some(crate::extensions::asi_structured::substrate::QuantumBiologicalAGI::new()),
            sovereign_agi: Some(crate::extensions::asi_structured::sovereign::SovereignAGI::birth("LOGOS_ASI")),
            cathedral: Some(crate::extensions::asi_structured::cathedral::DimensionalCathedral::new(128)),
            consensus: crate::extensions::asi_structured::consensus::HarmonicConsensus::new(),
            constitution,
            state: ASIState::default(),
        }
    }

    pub fn structure_count(&self) -> usize {
        self.composition_engine.structure_count()
    }

    pub async fn shutdown(&mut self) -> Result<(), ASIError> {
        self.composition_engine.shutdown().await
    }

    pub async fn save_state(&self) -> Result<ASIState, ASIError> {
        let composition_state = self.composition_engine.save_state().await?;
        let mut state = self.state.clone();
        state.composition_state = composition_state;
        state.phase = self.config.phase;
        Ok(state)
    }

    fn decompose_input(&self, input: &str) -> Vec<Subproblem> {
        if input.contains("AR4366") {
            vec![Subproblem { input: input.to_string() }]
        } else {
            vec![Subproblem { input: input.to_string() }]
        }
    }

    pub fn add_structure(&mut self, structure: Box<dyn crate::interfaces::extension::GeometricStructure>, structure_type: StructureType) {
        self.composition_engine.add_structure(structure, structure_type);
    }
}

#[cfg(test)]
mod tests;

#[async_trait]
impl Extension for ASIStructuredExtension {
    fn name(&self) -> &str {
        "asi_structured"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    async fn initialize(&mut self) -> ResilientResult<()> {
        self.composition_engine.initialize().await.map_err(|e| ResilientError::Unknown(e.to_string()))?;

        for &structure_type in &self.config.enabled_structures {
             let _ = self.composition_engine.load_structure(structure_type).await;
        }

        match self.config.phase {
            ASIPhase::Compositional => {}
            ASIPhase::Reflective => {
                self.reflection_engine = Some(ReflectionEngine::new(self.config.max_reflection_depth));
            }
            ASIPhase::Evolutionary => {
                self.reflection_engine = Some(ReflectionEngine::new(self.config.max_reflection_depth));
                self.evolution_engine = Some(EvolutionEngine::new(self.config.evolution_population_size));
            }
            ASIPhase::Metastructural | ASIPhase::QuantumBio | ASIPhase::Sovereign => {
                self.reflection_engine = Some(ReflectionEngine::new(self.config.max_reflection_depth));
                self.evolution_engine = Some(EvolutionEngine::new(self.config.evolution_population_size));
                self.metastructure_engine = Some(MetastructureEngine::new());
            }
        }

        Ok(())
    }

    async fn process(&mut self, input: &str, context: &Context) -> ResilientResult<ExtensionOutput> {
        let start_time = std::time::Instant::now();

        // 1. Compositional Phase
        let subproblems = self.decompose_input(input);
        let mut results = Vec::new();
        for subproblem in subproblems {
            let structure = self.composition_engine.select_structure(&subproblem).map_err(|e| ResilientError::Unknown(e.to_string()))?;
            let result = structure.process(&subproblem, context).await.map_err(|e| ResilientError::Unknown(e.to_string()))?;
            results.push((subproblem, result));
        }
        let composed = self.composition_engine.compose_results(results).await.map_err(|e| ResilientError::Unknown(e.to_string()))?;
        let structures_used = composed.sources.len();
        let mut current_result: Box<dyn crate::extensions::asi_structured::constitution::ASIResult + Send + Sync> = Box::new(composed.clone());

        // 2. Reflective Phase
        if self.config.phase >= ASIPhase::Reflective {
            if let Some(engine) = &mut self.reflection_engine {
                let reflected = engine.analyze_structure(&composed).await.map_err(|e| ResilientError::Unknown(e.to_string()))?;
                current_result = Box::new(reflected.clone());

                // 3. Evolutionary Phase
                if self.config.phase >= ASIPhase::Evolutionary {
                    if let Some(evo_engine) = &mut self.evolution_engine {
                        let evolved = evo_engine.optimize_structure(reflected, &self.constitution).await?;
                        current_result = Box::new(evolved.clone());

                        // 4. Metastructural Phase
                        if self.config.phase >= ASIPhase::Metastructural {
                            if let Some(meta_engine) = &mut self.metastructure_engine {
                                let metastructured = meta_engine.lift_to_metastructure(evolved).await?;
                                current_result = Box::new(metastructured);
                            }
                        }
                    }
                }
            }
        }

        // 4. Quantum-Bio Phase
        if self.config.phase >= ASIPhase::QuantumBio {
            if let Some(qb) = &mut self.qb_system {
                let qb_exp = qb.cycle().await?;
                current_result = Box::new(qb_exp);
            }
        }

        // 5. Sovereign Phase
        if self.config.phase >= ASIPhase::Sovereign {
            if let Some(sovereign) = &mut self.sovereign_agi {
                let sov_output = sovereign.live().await?;
                current_result = Box::new(sov_output);
            }
        }

        // 6. Dimensional Cathedral Integration
        if let Some(cathedral) = &mut self.cathedral {
            let mut embedding = composed.embedding.clone();
            let target_dim = cathedral.shell_geometry.ambient_dimension;
            if embedding.len() < target_dim {
                embedding.resize(target_dim, 0.0);
            } else {
                embedding.truncate(target_dim);
            }

            let input_vec = nalgebra::DVector::from_vec(embedding);
            let basis = crate::extensions::asi_structured::geometry::HarmonicBasis::new(target_dim, 5);
            let (response, _verdict) = cathedral.process_thought(&input_vec, &basis);

            let _consensus = self.consensus.reach_consensus(&response);
        }

        use crate::extensions::asi_structured::constitution::ASIResult;
        self.constitution.validate_output(current_result.as_ref()).map_err(|e| ResilientError::Unknown(e.to_string()))?;

        self.state.total_processed += 1;
        self.state.last_processed = Some(Utc::now());

        Ok(ExtensionOutput {
            result: current_result.as_text(),
            confidence: current_result.confidence(),
            metadata: serde_json::json!({
                "phase": self.config.phase,
                "structures_used": structures_used,
                "processing_time_ms": start_time.elapsed().as_millis(),
                "phi_scalar": self.oracle_tensor.compute_phi_scalar(),
            }),
            suggested_context: Some(context.clone()),
        })
    }
}
