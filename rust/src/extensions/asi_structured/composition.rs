use std::sync::Arc;
use std::collections::HashMap;

use super::{StructureType, ASIConstitution, CompositionStrategy};
use crate::interfaces::extension::{Subproblem, GeometricStructure};
use crate::extensions::asi_structured::structures::{StructureResult};
use crate::extensions::asi_structured::composer::{Composer, ComposedResult};
use crate::extensions::asi_structured::error::ASIError;

pub struct CompositionEngine {
    max_structures: usize,
    _active_strategy: CompositionStrategy,
    structures: HashMap<StructureType, Box<dyn GeometricStructure>>,
    composer: Composer,
    _constitution: Arc<ASIConstitution>,
}

impl CompositionEngine {
    pub fn new(
        max_structures: usize,
        default_strategy: CompositionStrategy,
        _constitution: Arc<ASIConstitution>,
    ) -> Self {
        Self {
            max_structures,
            _active_strategy: default_strategy,
            structures: HashMap::new(),
            composer: Composer::new(default_strategy),
            _constitution,
        }
    }

    pub async fn initialize(&mut self) -> Result<(), ASIError> {
        Ok(())
    }

    pub async fn load_structure(&mut self, structure_type: StructureType) -> Result<(), ASIError> {
        if self.structures.len() >= self.max_structures {
            return Err(ASIError::TooManyStructures {
                current: self.structures.len(),
                max: self.max_structures,
            });
        }

        let structure: Box<dyn GeometricStructure> = match structure_type {
            StructureType::TextEmbedding => {
                Box::new(crate::extensions::asi_structured::structures::text_embedding::TextEmbeddingStructure::new().await?)
            }
            StructureType::OracleDBA => {
                Box::new(crate::extensions::asi_structured::structures::oracle_dba::OracleDBAStructure::new())
            }
            StructureType::SequenceManifold => {
                Box::new(crate::extensions::asi_structured::structures::sequence_manifold::SequenceManifoldStructure::new())
            }
            StructureType::GraphComplex => {
                Box::new(crate::extensions::asi_structured::structures::graph_complex::GraphComplexStructure::new())
            }
            StructureType::HierarchicalSpace => {
                Box::new(crate::extensions::asi_structured::structures::hierarchical_space::HierarchicalSpaceStructure::new())
            }
            StructureType::TensorField => {
                Box::new(crate::extensions::asi_structured::structures::tensor_field::TensorFieldStructure::new())
            }
            StructureType::HPPP => {
                Box::new(crate::extensions::asi_structured::structures::hppp::HPPPStructure::new())
            }
            StructureType::SolarActivity => {
                Box::new(crate::extensions::asi_structured::solar::SolarActivityStructure::new("AR4366"))
            }
            _ => {
                return Ok(());
            }
        };

        self.structures.insert(structure_type, structure);
        Ok(())
    }

    pub fn add_structure(&mut self, structure: Box<dyn GeometricStructure>, structure_type: StructureType) {
        self.structures.insert(structure_type, structure);
    }

    pub fn select_structure(&self, subproblem: &Subproblem) -> Result<&dyn GeometricStructure, ASIError> {
        let mut best_score = -1.0;
        let mut best_structure = None;

        for structure in self.structures.values() {
            let score = structure.can_handle(subproblem);
            if score > best_score {
                best_score = score;
                best_structure = Some(structure.as_ref());
            }
        }

        if let Some(structure) = best_structure {
            if best_score > 0.0 {
                return Ok(structure);
            }
        }

        Err(ASIError::NoStructuresAvailable)
    }

    pub async fn compose_results(
        &self,
        results: Vec<(Subproblem, StructureResult)>,
    ) -> Result<ComposedResult, ASIError> {
        self.composer.compose(results).await
    }

    pub fn structure_count(&self) -> usize {
        self.structures.len()
    }

    pub async fn save_state(&self) -> Result<super::state::CompositionState, ASIError> {
        Ok(super::state::CompositionState::default())
    }

    pub async fn shutdown(&mut self) -> Result<(), ASIError> {
        self.structures.clear();
        Ok(())
use crate::interfaces::extension::{Context, Subproblem, StructureResult, Domain, GeometricStructure};
use crate::error::ResilientResult;
use crate::extensions::asi_structured::constitution::ASIResult;
use async_trait::async_trait;

pub struct CompositionEngine {
    pub structures: Vec<Box<dyn GeometricStructure>>,
    pub composer: StructureComposer,
}

impl CompositionEngine {
    pub fn new() -> Self {
        Self {
            structures: vec![],
            composer: StructureComposer { strategy: CompositionStrategy::Union, output_dim: 128 },
        }
    }

    pub async fn initialize(&mut self) -> ResilientResult<()> {
        Ok(())
    }

    pub fn add_structure(&mut self, structure: Box<dyn GeometricStructure>) {
        self.structures.push(structure);
    }

    pub fn select_structure(&self, subproblem: &Subproblem) -> ResilientResult<&dyn GeometricStructure> {
        let mut best_score = 0.0;
        let mut best_idx = 0;

        for (idx, structure) in self.structures.iter().enumerate() {
            let score = structure.can_handle(subproblem);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        if best_score < 0.3 {
            if self.structures.is_empty() {
                return Err(crate::error::ResilientError::Unknown("No structures available".into()));
            }
            return Ok(&*self.structures[0]);
        }

        Ok(&*self.structures[best_idx])
    }

    pub async fn compose_results(&self, results: Vec<(Subproblem, StructureResult)>) -> ResilientResult<ComposedResult> {
        match self.composer.strategy {
            CompositionStrategy::Union => self.compose_union(results),
            CompositionStrategy::Intersection => self.compose_intersection(results),
            CompositionStrategy::Sequence => self.compose_sequence(results).await,
            CompositionStrategy::Hierarchical => self.compose_hierarchical(results).await,
        }
    }

    fn compose_union(&self, results: Vec<(Subproblem, StructureResult)>) -> ResilientResult<ComposedResult> {
        let mut combined_embedding = vec![0.0; self.composer.output_dim];
        let mut total_weight = 0.0;

        for (_, result) in &results {
            let weight = result.confidence;
            for (i, val) in result.embedding.iter().enumerate() {
                if i < combined_embedding.len() {
                    combined_embedding[i] += val * weight;
                }
            }
            total_weight += weight;
        }

        if total_weight > 0.0 {
            for val in &mut combined_embedding {
                *val /= total_weight;
            }
        }

        Ok(ComposedResult {
            embedding: combined_embedding,
            sources: results.iter().map(|(s, _)| s.clone()).collect(),
            confidence: if results.is_empty() { 0.0 } else { total_weight / results.len() as f64 },
            composition_type: CompositionStrategy::Union,
        })
    }

    fn compose_intersection(&self, _results: Vec<(Subproblem, StructureResult)>) -> ResilientResult<ComposedResult> {
        Ok(ComposedResult::default())
    }

    async fn compose_sequence(&self, _results: Vec<(Subproblem, StructureResult)>) -> ResilientResult<ComposedResult> {
        Ok(ComposedResult::default())
    }

    async fn compose_hierarchical(&self, _results: Vec<(Subproblem, StructureResult)>) -> ResilientResult<ComposedResult> {
        Ok(ComposedResult::default())
    }

    pub fn active_structures_count(&self) -> usize {
        self.structures.len()
    }
}

pub struct StructureComposer {
    pub strategy: CompositionStrategy,
    pub output_dim: usize,
}

#[derive(Debug, Clone)]
pub enum CompositionStrategy {
    Union,
    Intersection,
    Sequence,
    Hierarchical,
}

#[derive(Debug, Clone, Default)]
pub struct ComposedResult {
    pub embedding: Vec<f64>,
    pub sources: Vec<Subproblem>,
    pub confidence: f64,
    pub composition_type: CompositionStrategy,
}

impl Default for CompositionStrategy {
    fn default() -> Self {
        CompositionStrategy::Union
    }
}

impl ASIResult for ComposedResult {
    fn to_string(&self) -> String {
        format!("ComposedResult(confidence: {:.2}, sources: {})", self.confidence, self.sources.len())
    }
    fn confidence(&self) -> f64 {
        self.confidence
    }
}
