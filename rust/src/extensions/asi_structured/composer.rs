use super::structures::StructureResult;
use crate::interfaces::extension::Subproblem;
use super::CompositionStrategy;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedResult {
    pub embedding: Vec<f64>,
    pub sources: Vec<String>,
    pub confidence: f64,
    pub strategy: CompositionStrategy,
    pub processing_time_ms: u128,
    pub metadata: serde_json::Value,
}

pub struct Composer {
    strategy: CompositionStrategy,
}

impl Composer {
    pub fn new(strategy: CompositionStrategy) -> Self {
        Self { strategy }
    }

    pub async fn compose(
        &self,
        results: Vec<(Subproblem, StructureResult)>,
    ) -> Result<ComposedResult, crate::extensions::asi_structured::error::ASIError> {
        if results.is_empty() {
            return Err(crate::extensions::asi_structured::error::ASIError::EmptyCompositionInput);
        }

        if results.len() == 1 {
            let (subproblem, result) = &results[0];
            return Ok(ComposedResult {
                embedding: result.embedding.clone(),
                sources: vec![subproblem.input.clone()],
                confidence: result.confidence,
                strategy: self.strategy,
                processing_time_ms: result.processing_time_ms,
                metadata: serde_json::json!({
                    "composition_type": "single_result",
                    "source_structure": result.source_structure_name,
                }),
            });
        }

        // Weighted Average composition
        let dim = results[0].1.embedding.len();
        let mut combined_embedding = vec![0.0; dim];
        let mut total_confidence = 0.0;
        let mut total_time = 0;
        let mut sources = Vec::new();

        for (subproblem, result) in &results {
            let weight = result.confidence;
            for (i, &val) in result.embedding.iter().enumerate() {
                if i < dim {
                    combined_embedding[i] += val * weight;
                }
            }
            total_confidence += weight;
            total_time += result.processing_time_ms;
            sources.push(subproblem.input.clone());
        }

        if total_confidence > 0.0 {
            for val in &mut combined_embedding {
                *val /= total_confidence;
            }
        }

        Ok(ComposedResult {
            embedding: combined_embedding,
            sources,
            confidence: total_confidence / results.len() as f64,
            strategy: self.strategy,
            processing_time_ms: total_time / results.len() as u128,
            metadata: serde_json::json!({
                "composition_type": "weighted_average",
                "source_count": results.len(),
            }),
        })
    }
}

impl ComposedResult {
    pub fn to_string(&self) -> String {
        format!(
            "Composed from {} sources (confidence: {:.2})",
            self.sources.len(), self.confidence
        )
    }
}
