pub struct VillageXNeurotoxicityModel {
    pub methylmercury_concentration: f64, // ppb
    pub absorption_fraction: f64,
    pub methylation_factor: f64,
    pub biological_half_life: f64,
    pub distribution_volume: f64,
}

pub struct HealthImpactProjection {
    pub segment: String,
    pub risk_level: String,
    pub primary_impact: String,
    pub severity: String,
}

impl VillageXNeurotoxicityModel {
    pub fn new_default() -> Self {
        Self {
            methylmercury_concentration: 0.42,
            absorption_fraction: 0.95,
            methylation_factor: 0.05,
            biological_half_life: 70.0, // days
            distribution_volume: 0.05, // L/kg
        }
    }

    pub fn calculate_blood_concentration(&self, fish_intake: f64, time: f64) -> f64 {
        let dose = fish_intake * self.methylmercury_concentration;
        let k = 0.693 / self.biological_half_life;
        (dose * self.absorption_fraction * self.methylation_factor / (k * self.distribution_volume)) * (1.0 - (-(k * time)).exp())
    }

    pub fn get_vulnerability_matrix() -> Vec<HealthImpactProjection> {
        vec![
            HealthImpactProjection {
                segment: "Gestantes / Fetos".to_string(),
                risk_level: "EXTREMO".to_string(),
                primary_impact: "Danos permanentes ao desenvolvimento do SNC".to_string(),
                severity: "Crítico".to_string(),
            },
            HealthImpactProjection {
                segment: "Crianças < 5 anos".to_string(),
                risk_level: "ALTO".to_string(),
                primary_impact: "Atrasos cognitivos, déficit de atenção, perda motora".to_string(),
                severity: "Elevado".to_string(),
            },
            HealthImpactProjection {
                segment: "Adultos".to_string(),
                risk_level: "MODERADO".to_string(),
                primary_impact: "Parestesia, redução do campo visual, fadiga crônica".to_string(),
                severity: "Sistêmico".to_string(),
            },
            HealthImpactProjection {
                segment: "Idosos".to_string(),
                risk_level: "MODERADO".to_string(),
                primary_impact: "Exacerbação de neurodegeneração preexistente".to_string(),
                severity: "Progressivo".to_string(),
            },
        ]
    }
}
