//! harmonia/src/soul/geometric_algebra.rs
//! Álgebra para operações unificadas no espaço semântico-geométrico

use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct GeometricVector {
    pub components: Array1<f64>,
    pub domain: String,
}

impl GeometricVector {
    pub fn new(components: Vec<f64>, domain: &str) -> Self {
        Self {
            components: Array1::from_vec(components),
            domain: domain.to_string(),
        }
    }

    pub fn to_unified_space(&self) -> Self {
        let mut matrices = HashMap::new();

        matrices.insert("concordia", Array2::from_shape_vec((3, 3), vec![0.8, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7]).unwrap());
        matrices.insert("sylva", Array2::from_shape_vec((3, 3), vec![0.1, 0.8, 0.1, 0.1, 0.2, 0.7, 0.3, 0.5, 0.2]).unwrap());
        matrices.insert("synesis", Array2::from_shape_vec((3, 3), vec![0.2, 0.1, 0.7, 0.6, 0.2, 0.2, 0.2, 0.3, 0.5]).unwrap());

        let transform = matrices.get(self.domain.as_str()).cloned().unwrap_or_else(|| Array2::eye(3));
        let new_components = self.components.dot(&transform);

        GeometricVector {
            components: new_components,
            domain: "harmonia_unified".to_string(),
        }
    }
}

pub fn geometric_product(v1: &GeometricVector, v2: &GeometricVector) -> GeometricVector {
    // Produto externo simplificado para capturar interações
    let n = v1.components.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        result[i] = v1.components[i] * v2.components[i];
    }

    // Adicionar termo de tensão (inter-relação)
    let tension = (v1.components.sum() * v2.components.sum()) * 0.1;
    result = result + tension;

    GeometricVector {
        components: result,
        domain: "harmonia_unified".to_string(),
    }
}

pub fn calculate_simplex_stability(vertices: &[GeometricVector]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }

    let unified: Vec<GeometricVector> = vertices.iter().map(|v| v.to_unified_space()).collect();
    let mut tensions = Vec::new();

    for i in 0..unified.len() {
        for j in i+1..unified.len() {
            let dot = unified[i].components.dot(&unified[j].components);
            let norm_i = unified[i].components.dot(&unified[i].components).sqrt();
            let norm_j = unified[j].components.dot(&unified[j].components).sqrt();

            let cos_sim = dot / (norm_i * norm_j);
            let tension = 1.0 - cos_sim.abs();
            tensions.push(tension);
        }
    }

    let avg_tension = if tensions.is_empty() { 1.0 } else { tensions.iter().sum::<f64>() / tensions.len() as f64 };
    1.0 / (1.0 + avg_tension)
}
