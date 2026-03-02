use std::collections::HashMap;
use num_complex::Complex64;
use nalgebra::{DMatrix, DVector};
use rustfft::{FftPlanner, num_complex::Complex as FftComplex};

pub type ConceptHash = [u8; 32];

pub struct HolographicGBS {
    pub concepts: HashMap<ConceptHash, DVector<Complex64>>,
    pub association_matrix: DMatrix<Complex64>,
    pub fft_planner: FftPlanner<f64>,
}

impl HolographicGBS {
    pub fn new(dim: usize) -> Self {
        Self {
            concepts: HashMap::new(),
            association_matrix: DMatrix::from_element(dim, dim, Complex64::new(0.0, 0.0)),
            fft_planner: FftPlanner::new(),
        }
    }

    pub fn store_concept(&mut self, hash: ConceptHash, vector: DVector<Complex64>) {
        let vector_conj = vector.map(|z| z.conj());
        let outer_product = &vector * vector_conj.transpose();
        self.association_matrix += outer_product;
        self.concepts.insert(hash, vector);
    }

    pub fn associative_recall(&self, partial_pattern: &DVector<Complex64>) -> Option<ConceptHash> {
        let recalled = &self.association_matrix * partial_pattern;
        self.concepts.iter()
            .max_by(|a, b| {
                let sim_a = a.1.dot(&recalled).norm();
                let sim_b = b.1.dot(&recalled).norm();
                sim_a.partial_cmp(&sim_b).unwrap()
            })
            .map(|(hash, _)| *hash)
    }

    /// Associação via multiplicação polinomial (convolução circular via FFT)
    /// Implementado conforme solicitado para atingir O(n log n)
    pub fn bind_concepts(&mut self, ψ_a: &DVector<Complex64>, ψ_b: &DVector<Complex64>) -> DVector<Complex64> {
        let n = ψ_a.len();
        let fft = self.fft_planner.plan_fft_forward(n);

        let mut a_fft: Vec<FftComplex<f64>> = ψ_a.iter().map(|z| FftComplex::new(z.re, z.im)).collect();
        let mut b_fft: Vec<FftComplex<f64>> = ψ_b.iter().map(|z| FftComplex::new(z.re, z.im)).collect();

        fft.process(&mut a_fft);
        fft.process(&mut b_fft);

        let mut prod_fft: Vec<FftComplex<f64>> = a_fft.iter().zip(b_fft.iter())
            .map(|(a, b)| a * b)
            .collect();

        let ifft = self.fft_planner.plan_fft_inverse(n);
        ifft.process(&mut prod_fft);

        // Normalização
        let result_vec: Vec<Complex64> = prod_fft.iter()
            .map(|z| Complex64::new(z.re, z.im) / (n as f64).sqrt())
            .collect();

        DVector::from_vec(result_vec)
    }
}
