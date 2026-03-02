use nalgebra::{DMatrix, Complex};
use num_complex::Complex64;

pub struct ComplexInferenceEngine {
    /// Pesos complexos: [layer, in, out] - simplificado para uma camada aqui
    pub weights: DMatrix<Complex64>,
    pub biases: DMatrix<Complex64>,
}

impl ComplexInferenceEngine {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            weights: DMatrix::from_element(rows, cols, Complex64::new(1.0, 0.0)),
            biases: DMatrix::from_element(rows, 1, Complex64::new(0.0, 0.0)),
        }
    }

    /// Nova função de ativação complexa (modReLU)
    pub fn complex_relu(z: Complex64) -> Complex64 {
        if z.norm() > 0.0 {
            // Mantém fase, ReLU na magnitude
            z / z.norm() * z.norm().max(0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    pub fn forward(&self, input: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let linear = &self.weights * input + &self.biases;
        linear.map(|z| Self::complex_relu(z))
    }
}
