use async_trait::async_trait;
use sha2::{Sha256, Digest};
use crate::interfaces::extension::{Context, Domain, Subproblem, StructureResult, GeometricStructure};
use crate::extensions::asi_structured::{StructureType};
use crate::error::{ResilientResult};

pub struct TextEmbeddingStructure;

impl TextEmbeddingStructure {
    pub async fn new() -> Result<Self, crate::extensions::asi_structured::error::ASIError> {
        Ok(Self)
    }

    fn generate_embedding(&self, text: &str) -> Vec<f64> {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let hash = hasher.finalize();

        let mut embedding = Vec::with_capacity(8);
        for i in 0..8 {
            let byte = hash[i % hash.len()];
            let value = (byte as f64 / 255.0) * 2.0 - 1.0;
            embedding.push(value);
        }

        embedding
    }
}

#[async_trait]
impl GeometricStructure for TextEmbeddingStructure {
    async fn process(
        &self,
        input: &Subproblem,
        _context: &Context,
    ) -> ResilientResult<StructureResult> {
        let start_time = std::time::Instant::now();

        let embedding = self.generate_embedding(&input.input);
        let processing_time = start_time.elapsed();

        Ok(StructureResult {
            embedding,
            confidence: 0.9,
            metadata: serde_json::json!({
                "model": "sha256-hash-based",
                "embedding_dim": 8,
            }),
            processing_time_ms: processing_time.as_millis(),
            source_structure_name: self.name().to_string(),
        })
    }

    fn can_handle(&self, _input: &Subproblem) -> f64 {
        0.9
    }

    fn domain(&self) -> Domain {
        Domain::Text
    }

    fn name(&self) -> &str {
        "text_embedding"
    }
}
