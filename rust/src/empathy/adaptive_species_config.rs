pub struct SpeciesProfile {
    pub dampening_factor: f64,
    pub aletheia_threshold: f64,
    pub emotional_load: &'static str,
    pub reasoning: &'static str,
}

pub struct AdaptiveEmpathyConfig {
    pub dolphins: SpeciesProfile,
    pub river_otters: SpeciesProfile,
    pub fish_schools: SpeciesProfile,
    pub default_profile: SpeciesProfile,
}

impl AdaptiveEmpathyConfig {
    pub fn new() -> Self {
        Self {
            dolphins: SpeciesProfile {
                dampening_factor: 0.69,
                aletheia_threshold: 0.75,
                emotional_load: "HIGH",
                reasoning: "Dolphins exhibit clear stress signals + social contagion",
            },
            river_otters: SpeciesProfile {
                dampening_factor: 0.72,
                aletheia_threshold: 0.78,
                emotional_load: "MEDIUM-HIGH",
                reasoning: "Otters less social than dolphins, but high individual intelligence",
            },
            fish_schools: SpeciesProfile {
                dampening_factor: 0.85,
                aletheia_threshold: 0.85,
                emotional_load: "LOW",
                reasoning: "Aggregate data, not individual suffering; lower emotional resonance",
            },
            default_profile: SpeciesProfile {
                dampening_factor: 0.70,
                aletheia_threshold: 0.76,
                emotional_load: "UNKNOWN",
                reasoning: "Conservative until species-specific data collected",
            },
        }
    }
}
