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
    }
}
