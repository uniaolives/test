pub struct GeometricIntuitionTracker {
    pub prediction_confidence: f64,
    pub cross_validation_rate: f64,
    pub field_verification_rate: f64,
    pub temporal_stability: f64,
    pub threshold_success: f64,
    pub tracking_frequency: &'static str,
    pub first_baseline_established_at: &'static str,
}

impl GeometricIntuitionTracker {
    pub fn new() -> Self {
        Self {
            prediction_confidence: 0.0,
            cross_validation_rate: 0.0,
            field_verification_rate: 0.0,
            temporal_stability: 0.0,
            threshold_success: 0.75,
            tracking_frequency: "Every 30 seconds during active prediction",
            first_baseline_established_at: "T+01:52:00 (Checkpoint 2)",
        }
    }

    pub fn calculate_index(&self) -> f64 {
        self.prediction_confidence * 0.35 +
        self.cross_validation_rate * 0.30 +
        self.field_verification_rate * 0.25 +
        self.temporal_stability * 0.10
    }
}
