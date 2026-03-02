// rust/src/microtubule_biology.rs
// Real Microtubule Biology implementation based on established biophysics.

use chrono::{DateTime, Utc};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct RealMicrotubule {
    pub length: f64,           // Comprimento em micrômetros (μm)
    pub diameter: f64,         // Diâmetro em nanômetros (25nm típico)
    pub tubulin_dimers: u64,   // Número de dímeros de tubulina
    pub gtp_bound: bool,       // GTP ligado (crescimento) vs GDP (encolhimento)
    pub ptm_modifications: Vec<PTM>,
    pub dynamic_instability: DynamicInstability,
    pub bending_stiffness: f64, // Rigidez à flexão (pN·μm²)
    pub persistence_length: f64, // Comprimento de persistência (~5200μm)
    pub youngs_modulus: f64,    // Módulo de Young (~1.2 GPa)
}

#[derive(Debug, Clone)]
pub struct DynamicInstability {
    pub growth_phase: bool,
    pub growth_rate: f64,       // Velocidade de crescimento (μm/min)
    pub shrinkage_rate: f64,    // Velocidade de encolhimento (μm/min)
    pub catastrophe_rate: f64,  // Taxa de catástrofe (1/min)
    pub rescue_rate: f64,       // Taxa de resgate (1/min)
}

#[derive(Debug, Clone)]
pub enum PTM {
    Acetylation,
    Tyrosination,
    Detyrosination,
    Polyglutamylation,
    Polyglycylation,
}

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub final_length: f64,
    pub phase: String,
    pub gtp_cap_status: bool,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct MechanicalProperties {
    pub flexural_rigidity: f64,    // pN·μm²
    pub persistence_length: f64,   // μm
    pub critical_buckling_force: f64, // pN
}

impl RealMicrotubule {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            length: rng.gen_range(1.0..6.0),
            diameter: 25.0,
            tubulin_dimers: 13 * 100, // Approximate
            gtp_bound: true,
            ptm_modifications: vec![PTM::Acetylation],
            dynamic_instability: DynamicInstability {
                growth_phase: true,
                growth_rate: rng.gen_range(1.0..6.0),
                shrinkage_rate: rng.gen_range(5.0..15.0),
                catastrophe_rate: 0.25,
                rescue_rate: 0.15,
            },
            bending_stiffness: 22.0,
            persistence_length: 5200.0,
            youngs_modulus: 1.2,
        }
    }

    pub fn simulate_dynamics(&mut self, time_step_min: f64, _gtp_concentration: f64) -> SimulationResult {
        let mut rng = rand::thread_rng();

        if self.gtp_bound {
            // FASE DE CRESCIMENTO
            let growth_length = self.dynamic_instability.growth_rate * time_step_min;
            self.length += growth_length;

            // Chance de catástrofe
            if rng.gen::<f64>() < self.dynamic_instability.catastrophe_rate * time_step_min {
                self.gtp_bound = false;
                self.dynamic_instability.growth_phase = false;
            }
        } else {
            // FASE DE ENCOLHIMENTO
            let shrinkage_length = self.dynamic_instability.shrinkage_rate * time_step_min;
            self.length = (self.length - shrinkage_length).max(0.0);

            // Chance de resgate
            if rng.gen::<f64>() < self.dynamic_instability.rescue_rate * time_step_min {
                self.gtp_bound = true;
                self.dynamic_instability.growth_phase = true;
            }
        }

        SimulationResult {
            final_length: self.length,
            phase: if self.gtp_bound { "Growing" } else { "Shrinking" }.to_string(),
            gtp_cap_status: self.gtp_bound,
            timestamp: Utc::now(),
        }
    }

    pub fn calculate_mechanical_properties(&self) -> MechanicalProperties {
        let flexural_rigidity = self.bending_stiffness;
        let thermal_energy = 4.1e-3; // pN·μm at 37°C (approx)
        let persistence_length = flexural_rigidity / thermal_energy;

        MechanicalProperties {
            flexural_rigidity,
            persistence_length,
            critical_buckling_force: self.calculate_buckling_force(),
        }
    }

    fn calculate_buckling_force(&self) -> f64 {
        let youngs_modulus_pa = self.youngs_modulus * 1e9;
        let r_outer = self.diameter * 1e-9 / 2.0;
        let r_inner = (self.diameter - 14.0) * 1e-9 / 2.0;
        let moment_of_inertia = std::f64::consts::PI * (r_outer.powi(4) - r_inner.powi(4)) / 4.0;
        let length_m = self.length * 1e-6;

        if length_m > 0.0 {
            (std::f64::consts::PI.powi(2) * youngs_modulus_pa * moment_of_inertia / length_m.powi(2)) * 1e12 // Convert to pN
        } else {
            0.0
        }
    }
}
