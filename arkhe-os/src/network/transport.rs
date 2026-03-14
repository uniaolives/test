// arkhe-os/src/network/transport.rs

use reed_solomon_erasure::galois_8::ReedSolomon;
use std::sync::Arc;

/// Configuração do Tzinor de Dados
pub struct TzinorConfig {
    pub data_shards: usize,
    pub parity_shards: usize,
    pub total_shards: usize,
}

impl Default for TzinorConfig {
    fn default() -> Self {
        Self {
            data_shards: 16,
            parity_shards: 8,
            total_shards: 24,
        }
    }
}

/// O Transportador do Tzinor
pub struct TzinorTransport {
    config: TzinorConfig,
    codec: ReedSolomon,
}

impl TzinorTransport {
    pub fn new(config: TzinorConfig) -> Self {
        let codec = ReedSolomon::new(config.data_shards, config.parity_shards)
            .expect("Failed to create Reed-Solomon codec");
        Self { config, codec }
    }

    pub fn shard_data(&self, data: &[u8]) -> Vec<Vec<u8>> {
        let chunk_size = (data.len() + self.config.data_shards - 1) / self.config.data_shards;
        let mut shards: Vec<Vec<u8>> = vec![vec![0; chunk_size]; self.config.total_shards];

        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            shards[i][..chunk.len()].copy_from_slice(chunk);
        }

        self.codec.encode(&mut shards).expect("Encoding failed");
        shards
    }

    pub fn reconstruct_data(&self, mut shards: Vec<Option<Vec<u8>>>) -> Result<Vec<u8>, String> {
        self.codec.reconstruct(&mut shards).map_err(|e| format!("{:?}", e))?;

        let mut result = Vec::new();
        for i in 0..self.config.data_shards {
            if let Some(ref shard) = shards[i] {
                result.extend_from_slice(shard);
            }
        }
        Ok(result)
    }
}
