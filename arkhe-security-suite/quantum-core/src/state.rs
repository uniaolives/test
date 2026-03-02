#[derive(Clone, Debug)]
pub struct QuantumState {
    pub density_matrix: Vec<Vec<f64>>,
    pub entropy: f64,
}

impl QuantumState {
    pub fn new_pure() -> Self {
        Self {
            density_matrix: vec![vec![1.0, 0.0], vec![0.0, 0.0]],
            entropy: 0.0,
        }
    }
}
