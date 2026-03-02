pub struct Gradient {
    pub magnitude: f64,
}

pub struct VariationalFreeEnergy {
    pub expected_energy: f64,
    pub entropy: f64,
    pub temperature: f64,
}

impl VariationalFreeEnergy {
    pub fn value(&self) -> f64 {
        self.expected_energy - self.temperature * self.entropy
    }

    pub fn compute(observed: &crate::manifold::QuantumState, model: &crate::asi_core::InternalModel) -> Self {
        let prediction_error = observed.surprise_given(model);
        VariationalFreeEnergy {
            expected_energy: prediction_error,
            entropy: model.entropy,
            temperature: 300.0,
        }
    }

    pub fn gradient(&self) -> Gradient {
        Gradient {
            magnitude: self.value().abs(),
        }
    }
}
