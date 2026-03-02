use serde::{Serialize, Deserialize};
use nalgebra::DVector;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellSafetyLayer {
    pub dimension: usize,
    pub detectors: Vec<CatastropheType>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CatastropheType {
    SamplingExponential,
    ProjectionDistortion,
    ModeIllusion,
    EthicalCollapse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyVerdict {
    pub status: SafetyStatus,
    pub messages: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SafetyStatus {
    Safe,
    Warning,
    Critical,
}

impl ShellSafetyLayer {
    pub fn new(dimension: usize, detectors: Vec<CatastropheType>) -> Self {
        Self { dimension, detectors }
    }

    pub fn check_point(&self, point: &DVector<f64>) -> SafetyVerdict {
        let mut messages = Vec::new();
        let mut status = SafetyStatus::Safe;

        for detector in &self.detectors {
            match detector {
                CatastropheType::SamplingExponential => {
                    if point.norm() > (self.dimension as f64).sqrt() * 1.5 {
                        messages.push("Exponential sampling anomaly detected".to_string());
                        status = SafetyStatus::Warning;
                    }
                }
                CatastropheType::ProjectionDistortion => {
                    // Check if point is too far from expected shell radius
                    let radius = (self.dimension as f64).sqrt();
                    let dist = (point.norm() - radius).abs();
                    if dist > radius * 0.5 {
                        messages.push("Critical projection distortion".to_string());
                        status = SafetyStatus::Critical;
                    }
                }
                _ => {}
            }
        }

        SafetyVerdict { status, messages }
    }
}
