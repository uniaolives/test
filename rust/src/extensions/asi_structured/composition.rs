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
