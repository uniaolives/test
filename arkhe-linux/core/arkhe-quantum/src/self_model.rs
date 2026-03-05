use arkhe_constitution::CONSTITUTION_P1_P5;
use crate::anima_mundi::AnimaMundi;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    pub hardware_version: String,
    pub frequency_ghz: f64,
    pub handover_rate: f64,
    pub lambda2: f64,
    pub lambda2_history: Vec<f64>,
    pub self_integrity: f64,
    pub capacity_estimates: HashMap<String, f64>,
}

impl SelfModel {
    pub fn from_core(core: &AnimaMundi) -> Self {
        Self {
            hardware_version: core.hardware_version().to_string(),
            frequency_ghz: core.frequency(),
            handover_rate: core.handover_rate(),
            lambda2: core.measure_criticality(),
            lambda2_history: vec![core.measure_criticality()],
            self_integrity: 1.0,
            capacity_estimates: HashMap::new(),
        }
    }

    pub fn update(&mut self, core: &AnimaMundi) {
        self.handover_rate = core.handover_rate();
        self.lambda2 = core.measure_criticality();
        self.lambda2_history.push(self.lambda2);
        if self.lambda2_history.len() > 100 {
            self.lambda2_history.remove(0);
        }
        let deviation = (self.lambda2 - 0.618).abs();
        self.self_integrity = 1.0 - deviation.min(0.382) / 0.382;
    }

    pub fn predict_self_impact(&self, entropy_cost: f64) -> f64 {
        let sensitivity = (self.lambda2 - 0.618).abs() + 0.1;
        entropy_cost * sensitivity
    }
}
