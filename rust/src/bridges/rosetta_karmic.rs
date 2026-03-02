use crate::ethics::karmic::core::SoulState;

pub struct ConstitutionalInsight {
    pub phi_equivalent: f64,
    pub curvature_risk: bool,
    pub recommendation: String,
}

pub struct RosettaKarmicBridge;

impl RosettaKarmicBridge {
    pub fn translate_to_constitutional(
        karmic_state: &SoulState,
    ) -> Result<ConstitutionalInsight, String> {

        if karmic_state.constitutional_influence != 0.0 {
            return Err("VIOLACAO: Influencia constitucional detectada".to_string());
        }

        let grade_val = karmic_state.grade.value();
        let phi_equivalent = match grade_val {
            1..=24 => 0.60 + (grade_val as f64 * 0.002),
            25..=49 => 0.65 + ((grade_val - 25) as f64 * 0.002),
            50..=74 => 0.70 + ((grade_val - 50) as f64 * 0.002),
            75..=98 => 0.75 + ((grade_val - 75) as f64 * 0.002),
            99 => 0.95,
            _ => return Err("Invalid Grade".to_string()),
        };

        Ok(ConstitutionalInsight {
            phi_equivalent,
            curvature_risk: karmic_state.service_ratio < 0.5,
            recommendation: "ObservationalOnly".to_string(),
        })
    }
}
