// star_node.rs
use nalgebra::Vector3;

/// Representa um nó STAR com capacidade de simulação de vórtice (circumvolution cissoid).
pub struct STARNode {
    pub position: Vector3<f64>,
    pub angular_momentum: Vector3<f64>,
    pub scalar_a: f64,           // Densidade de estresse EM (quaternion scalar)
    pub torsion: Vector3<f64>,   // Tensor de torção de Einstein-Cartan
}

impl STARNode {
    pub fn new(position: Vector3<f64>) -> Self {
        Self {
            position,
            angular_momentum: Vector3::new(0.0, 0.0, 0.0),
            scalar_a: 1.0,
            torsion: Vector3::new(0.0, 0.0, 0.0),
        }
    }

    /// Aplica torque de uma onda convergente, atualizando a torção do nó.
    /// Este é o mecanismo que converte convergência radial em espiral (Part III da GGF).
    pub fn apply_torque(&mut self, wave_vector: Vector3<f64>, intensity: f64) {
        // A torção é proporcional ao produto vetorial do vetor de onda com o momento angular
        self.torsion += wave_vector.cross(&self.angular_momentum) * intensity;

        // Atualiza o momento angular baseado na torção
        self.angular_momentum += self.torsion * 0.01;  // Fator de amortecimento
    }

    /// Retorna o índice de refração GRIN neste nó: n = scalar_a (proporcional à densidade)
    pub fn refractive_index(&self) -> f64 {
        self.scalar_a
    }
}
