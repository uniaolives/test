pub struct MobileTreatmentLogistics {
    pub units_required: u32,
    pub technology: String,
    pub deployment_options: DeploymentOptions,
    pub operational_parameters: OperationalParameters,
}

pub struct DeploymentOptions {
    pub air_drop: OptionDetails,
    pub river_transport: OptionDetails,
    pub hybrid: OptionDetails,
}

pub struct OptionDetails {
    pub eta: String,
    pub risk: String,
    pub feasibility: f64,
}

pub struct OperationalParameters {
    pub power_source: String,
    pub maintenance_requirement: String,
    pub effective_reduction: String,
}

impl MobileTreatmentLogistics {
    pub fn simulate_scenario_b() -> Self {
        Self {
            units_required: 4,
            technology: "Activated Carbon + Ion Exchange (Selective for MeHg)".to_string(),
            deployment_options: DeploymentOptions {
                air_drop: OptionDetails {
                    eta: "14 hours".to_string(),
                    risk: "Alto (integridade das membranas)".to_string(),
                    feasibility: 0.82,
                },
                river_transport: OptionDetails {
                    eta: "32 hours".to_string(),
                    risk: "Baixo".to_string(),
                    feasibility: 0.94,
                },
                hybrid: OptionDetails {
                    eta: "18 hours".to_string(),
                    risk: "MÍNIMO".to_string(),
                    feasibility: 0.96,
                },
            },
            operational_parameters: OperationalParameters {
                power_source: "Solar/Battery Hybrid (Integrated)".to_string(),
                maintenance_requirement: "Filtros trocados a cada 120h sob carga de 0.42 ppb".to_string(),
                effective_reduction: "92% de remoção de MeHg garantida".to_string(),
            },
        }
    }

    pub fn calculate_remediated_concentration(&self, dietary_dose: f64, elimination_k: f64, time: f64) -> f64 {
        let filter_efficiency = 0.921; // 92.1% removal
        let intake_rate = dietary_dose * (1.0 - filter_efficiency);
        // Integral of intake * e^(-k(t-tau)) d tau from 0 to t
        // Result: (intake / k) * (1 - e^(-kt))
        (intake_rate / elimination_k) * (1.0 - (-(elimination_k * time)).exp())
    }
}
