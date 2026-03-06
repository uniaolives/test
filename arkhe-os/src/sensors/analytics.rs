use std::collections::BTreeMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandMetric {
    pub timestamp_ns: u64,
    pub power: f64,
}

pub struct MultivariateKurtosis {
    pub buffers: BTreeMap<String, Vec<f64>>,
    pub capacity: usize,
}

impl MultivariateKurtosis {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffers: BTreeMap::new(),
            capacity,
        }
    }

    pub fn push(&mut self, band: &str, value: f64) {
        let buffer = self.buffers.entry(band.to_string()).or_insert_with(Vec::new);
        buffer.push(value);
        if buffer.len() > self.capacity {
            buffer.remove(0);
        }
    }

    pub fn calculate(&self) -> f64 {
        // Implementação simplificada de Kurtosis Multivariada (Mardia's Kurtosis)
        // Em um sistema real, calcularíamos a matriz de covariância e a forma quadrática
        let n_bands = self.buffers.len();
        if n_bands == 0 { return 0.0; }

        let mut total_kurtosis = 0.0;
        for buffer in self.buffers.values() {
            if buffer.len() < 4 { continue; }
            let mean = buffer.iter().sum::<f64>() / buffer.len() as f64;
            let m2 = buffer.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / buffer.len() as f64;
            let m4 = buffer.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / buffer.len() as f64;
            let k = m4 / (m2.powi(2) + 1e-10) - 3.0;
            total_kurtosis += k;
        }

        total_kurtosis / n_bands as f64
    }
}
