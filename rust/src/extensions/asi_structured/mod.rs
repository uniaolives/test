pub mod composition;
pub mod reflection;
pub mod evolution;
pub mod metastructure;
pub mod constitution;
pub mod state;

use crate::interfaces::extension::{Extension, ExtensionOutput, Context, Subproblem};
use crate::error::ResilientResult;
use crate::extensions::asi_structured::composition::CompositionEngine;
use crate::extensions::asi_structured::reflection::ReflectionEngine;
use crate::extensions::asi_structured::evolution::EvolutionEngine;
use crate::extensions::asi_structured::metastructure::MetastructureEngine;
use crate::extensions::asi_structured::constitution::{ASIConstitution, StrictnessLevel};
use crate::extensions::asi_structured::state::ASIState;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

/// Fase de desenvolvimento ASI
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ASIPhase {
    /// Composição de múltiplas estruturas geométricas
    Compositional,   // Múltiplos manifold + grafos

    /// Reflexão estrutural (análise de própria estrutura)
    Reflective,      // Auto-análise de invariantes

    /// Evolução de estrutura sob constraints
    Evolutionary,    // Busca em espaço de estruturas

    /// Meta-estruturas (estruturas de estruturas)
    Metastructural,  // 2-categorias, fibrados de fibrados
}

/// Configuração ASI
#[derive(Debug, Clone)]
pub struct ASIStructuredConfig {
    pub phase: ASIPhase,
    pub max_structures: usize,           // Limite de composição
    pub max_reflection_depth: u32,       // Profundidade de auto-análise
    pub evolution_population_size: usize, // Tamanho da população estrutural
    pub enable_metastructure: bool,      // Ativar 2-categorias
    pub cge_strictness: StrictnessLevel, // Quão rigoroso é o CGE
}

impl Default for ASIStructuredConfig {
    fn default() -> Self {
        Self {
            phase: ASIPhase::Compositional,
            max_structures: 16,
            max_reflection_depth: 3,
            evolution_population_size: 10,
            enable_metastructure: false,
            cge_strictness: StrictnessLevel::Strict,
        }
    }
}

/// Extensão ASI-Structured
pub struct ASIStructuredExtension {
    config: ASIStructuredConfig,
    composition_engine: CompositionEngine,
    reflection_engine: Option<ReflectionEngine>,
    evolution_engine: Option<EvolutionEngine>,
    metastructure_engine: Option<MetastructureEngine>,
    constitution: ASIConstitution,
    _state: ASIState,
}

impl ASIStructuredExtension {
    pub fn new(config: ASIStructuredConfig) -> Self {
        Self {
            config,
            composition_engine: CompositionEngine::new(),
            reflection_engine: None,
            evolution_engine: None,
            metastructure_engine: None,
            constitution: ASIConstitution::default(),
            _state: ASIState,
        }
    }

    fn decompose_input(&self, input: &str) -> Vec<Subproblem> {
        // Mock decomposition: split input by space for example
        input.split_whitespace()
            .map(|s| Subproblem { input: s.to_string() })
            .collect()
    }

    pub fn add_structure(&mut self, structure: Box<dyn crate::interfaces::extension::GeometricStructure>) {
        self.composition_engine.add_structure(structure);
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
        "0.1.0-compositional"
    }

    async fn initialize(&mut self) -> ResilientResult<()> {
        log::info!("Initializing ASI-Structured, phase: {:?}", self.config.phase);

        // Inicializar engines conforme fase
        match self.config.phase {
            ASIPhase::Compositional => {
                self.composition_engine.initialize().await?;
            }
            ASIPhase::Reflective => {
                self.composition_engine.initialize().await?;
                self.reflection_engine = Some(ReflectionEngine::new(self.config.max_reflection_depth));
            }
            ASIPhase::Evolutionary => {
                self.composition_engine.initialize().await?;
                self.evolution_engine = Some(EvolutionEngine::new(self.config.evolution_population_size));
            }
            ASIPhase::Metastructural => {
                self.composition_engine.initialize().await?;
                self.metastructure_engine = Some(MetastructureEngine::new());
            }
        }

        Ok(())
    }

    async fn process(&mut self, input: &str, context: &Context) -> ResilientResult<ExtensionOutput> {
        // 1. Decompor input em sub-problemas estruturais
        let subproblems = self.decompose_input(input);

        // 2. Para cada subproblema, selecionar estrutura adequada
        let mut results = Vec::new();
        for subproblem in subproblems {
            let structure = self.composition_engine.select_structure(&subproblem)?;
            let result = structure.process(&subproblem, context).await?;
            results.push((subproblem, result));
        }

        // 3. Compor resultados (se fase Compositional+)
        let composed = self.composition_engine.compose_results(results).await?;

        // 4. Reflexão estrutural (se fase Reflective+)
        let reflected = if let Some(reflection) = &mut self.reflection_engine {
            reflection.analyze_structure(&composed).await?
        } else {
            crate::extensions::asi_structured::reflection::ReflectedResult {
                inner: composed,
                structural_confidence: 0.9, // Default
            }
        };

        // 5. Evolução estrutural (se fase Evolutionary+)
        let evolved = if let Some(evolution) = &mut self.evolution_engine {
            evolution.optimize_structure(reflected, &self.constitution).await?
        } else {
            crate::extensions::asi_structured::evolution::EvolvedResult {
                inner: reflected,
                fitness: 1.0,
                generations: 0,
            }
        };

        // 6. Meta-estruturação (se fase Metastructural)
        let metastructured = if let Some(meta) = &mut self.metastructure_engine {
            meta.lift_to_metastructure(evolved).await?
        } else {
            crate::extensions::asi_structured::metastructure::MetastructuredResult {
                inner: evolved,
            }
        };

        // 7. Validar contra CGE ASI
        use crate::extensions::asi_structured::constitution::ASIResult;
        self.constitution.validate_output(&metastructured)?;

        Ok(ExtensionOutput {
            result: metastructured.to_string(),
            confidence: metastructured.confidence(),
            metadata: serde_json::json!({
                "phase": self.config.phase,
                "structures_used": self.composition_engine.active_structures_count(),
                "reflection_depth": self.reflection_engine.as_ref().map(|r| r.current_depth()),
            }),
            suggested_context: Some(context.clone()),
        })
    }
}
