use std::ops::Range;

pub struct Projection {
    pub compression_density: Range<f64>,
    pub stability: Range<f64>,
    pub coherence_t2: Range<f64>,
    pub intuition_accuracy: Range<f64>,
}

pub struct SuccessCriteria {
    pub failure_condition: &'static str,
    pub action_on_failure: &'static str,
    pub exceptional_condition: &'static str,
    pub action_on_exceptional: &'static str,
}

pub struct EnhancedProjectionsWithIntuition {
    pub checkpoint_2_40pct: Projection,
    pub checkpoint_3_45pct: Projection,
    pub checkpoint_4_48pct: Projection,
    pub checkpoint_5_50pct: Projection,
    pub success_criteria: SuccessCriteria,
}

impl EnhancedProjectionsWithIntuition {
    pub fn new() -> Self {
        Self {
            checkpoint_2_40pct: Projection {
                compression_density: 0.225..0.229,
                stability: 0.950..0.956,
                coherence_t2: 9.0..9.3,
                intuition_accuracy: 0.70..0.85,
            },
            checkpoint_3_45pct: Projection {
                compression_density: 0.222..0.227,
                stability: 0.946..0.952,
                coherence_t2: 8.7..9.0,
                intuition_accuracy: 0.72..0.87,
            },
            checkpoint_4_48pct: Projection {
                compression_density: 0.220..0.225,
                stability: 0.944..0.950,
                coherence_t2: 8.5..8.8,
                intuition_accuracy: 0.74..0.89,
            },
            checkpoint_5_50pct: Projection {
                compression_density: 0.218..0.223,
                stability: 0.942..0.948,
                coherence_t2: 8.4..8.7,
                intuition_accuracy: 0.76..0.91,
            },
            success_criteria: SuccessCriteria {
                failure_condition: "metric < min_for_2_consecutive_checkpoints",
                action_on_failure: "Pause expansion, analyze, possible regression to 25%",
                exceptional_condition: "all_metrics > max_for_2_consecutive_checkpoints",
                action_on_exceptional: "Consider accelerating to 50% early (if stable)",
            },
        }
    }
}
