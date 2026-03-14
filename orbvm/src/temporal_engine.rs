use ndarray::Array1;
use num_complex::Complex64;
use rustfft::{FftPlanner, num_complex::Complex};

pub struct TemporalEngine {
    pub coupling_k: f64,
}

impl TemporalEngine {
    pub fn new(k: f64) -> Self {
        Self { coupling_k: k }
    }

    /// Sincronização de Kuramoto simplificada
    pub fn synchronize(&self, phases: &mut Array1<f64>) -> f64 {
        let n = phases.len() as f64;
        let mut sum_sin = 0.0;
        let mut sum_cos = 0.0;

        for &phi in phases.iter() {
            sum_sin += phi.sin();
            sum_cos += phi.cos();
        }

        let r = (sum_sin.powi(2) + sum_cos.powi(2)).sqrt() / n;

        // Atualiza fases (Euler step simplificado)
        let mean_phase = sum_sin.atan2(sum_cos);
        for phi in phases.iter_mut() {
            *phi += self.coupling_k * (mean_phase - *phi).sin();
        }

        r // Retorna coerência λ₂
    }

    pub fn extract_eigenstates(&self, signal: &[f64]) -> Vec<f64> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(signal.len());

        let mut buffer: Vec<Complex<f64>> = signal
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut buffer);

        buffer.iter().map(|c| c.norm()).collect()
    }
}
