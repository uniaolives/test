use super::oracle_tensor::OracleTensorState;
use nalgebra::DVector;

pub struct QuenchingEngine {
    pub replicas: Vec<OracleTensorState>,
}

impl QuenchingEngine {
    pub fn new() -> Self {
        Self {
            replicas: vec![OracleTensorState::new(), OracleTensorState::new(), OracleTensorState::new()],
        }
    }

    pub async fn execute_quench(&mut self) -> QuenchResult {
        let phi_values: Vec<f64> = self.replicas
            .iter()
            .map(|r| r.compute_phi_scalar())
            .collect();

        let mut sorted_phi = phi_values.clone();
        sorted_phi.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let consensus_phi = sorted_phi[1]; // Medial 2-of-3

        let mut byzantine_nodes = Vec::new();
        for (i, &p) in phi_values.iter().enumerate() {
            if (p - consensus_phi).abs() > 0.1 {
                byzantine_nodes.push(i);
            }
        }

        QuenchResult {
            consensus_phi,
            byzantine_nodes,
        }
    }
}

pub struct QuenchResult {
    pub consensus_phi: f64,
    pub byzantine_nodes: Vec<usize>,
}
