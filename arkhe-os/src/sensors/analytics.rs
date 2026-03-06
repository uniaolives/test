use nalgebra::{DMatrix};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandMetric {
    pub timestamp_ns: u64,
    pub power: f64,
}

pub struct MultivariateAnalytics {
    pub buffers: std::collections::BTreeMap<String, Vec<f64>>,
    pub capacity: usize,
}

impl MultivariateAnalytics {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffers: std::collections::BTreeMap::new(),
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

    pub fn sample_kurtosis(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 4.0 { return 0.0; }
        let mean = data.iter().sum::<f64>() / n;
        let m2 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
        if m2 == 0.0 { return 0.0; }
        m4 / (m2.powi(2)) - 3.0
    }

    pub fn mardia_kurtosis(&self) -> f64 {
        let bands: Vec<&String> = self.buffers.keys().collect();
        let p = bands.len();
        if p == 0 { return 0.0; }

        let n = self.buffers.values().map(|v| v.len()).min().unwrap_or(0);
        if n < p + 2 { return 0.0; }

        let mut mat = DMatrix::zeros(n, p);
        for (j, band) in bands.iter().enumerate() {
            let samples = &self.buffers[*band];
            for i in 0..n {
                mat[(i, j)] = samples[samples.len() - n + i];
            }
        }

        for mut col in mat.column_iter_mut() {
            let mean = col.mean();
            for val in col.iter_mut() {
                *val -= mean;
            }
        }

        let cov = (&mat.transpose() * &mat) / (n as f64 - 1.0);
        let cov_inv = match cov.try_inverse() {
            Some(inv) => inv,
            None => return 0.0,
        };

        let mut d4_sum = 0.0;
        for i in 0..n {
            let row = mat.row(i);
            let d2 = (row * &cov_inv * row.transpose())[(0, 0)];
            d4_sum += d2 * d2;
        }

        (d4_sum / n as f64) - (p * (p + 2)) as f64
    }
}
