// src/triad/types.rs

#[derive(Clone, Debug)]
pub struct ConstitutionalState;

#[derive(Clone, Debug)]
pub struct ProposedAction;

impl ProposedAction {
    pub fn thermodynamic_cost(&self) -> f64 { 10.0 }
}

#[derive(Debug, Clone)]
pub enum Action {
    MandatoryVaccination,
    VoluntaryWithIncentives,
    PassportSanitaryRestricted,
    TotalLiberty,
}

impl Action {
    pub fn thermodynamic_cost(&self) -> f64 { 10.0 }
}

#[derive(Debug)]
pub enum ConstitutionalError {
    DignityViolation,
}

#[derive(Clone, Debug)]
pub struct FlourishingGradient {
    pub direction: Vec<f64>,
    pub magnitude: f64,
    pub constitutional_valid: bool,
}

pub struct FlourishingOutput;
impl FlourishingOutput {
    pub fn is_significant(&self) -> bool { true }
}

pub struct ConstitutionalResolution;

#[derive(Clone, Debug)]
pub struct ConstitutionalIdentity;

#[derive(Clone, Debug)]
pub struct ManifoldCurvature {
    pub ethical_curvature: f64,
    pub resolution_energy: f64,
    pub energy_required_to_flatten: f64,
    pub constitutional_implications: String,
}

#[derive(Clone, Debug)]
pub struct KarmicState {
    pub sto_sts_ratio: f64,
}

#[derive(PartialEq, Clone, Debug)]
pub enum Demand {
    Transparency,
}

impl ConstitutionalIdentity {
    pub fn is_preserved(&self, _components: &[crate::autopoiesis::organizational_closure::ConstitutionalComponent], _topology: &crate::autopoiesis::organizational_closure::ProductionTopology) -> bool { true }
    pub fn from_genesis() -> Self { ConstitutionalIdentity }
}

pub struct ConstitutionalInsight {
    pub recommendation: &'static str,
    pub phi_adjustment: f64,
    pub energy_budget_delta: f64,
    pub reason: &'static str,
}

impl Default for ConstitutionalInsight {
    fn default() -> Self {
        Self {
            recommendation: "",
            phi_adjustment: 0.0,
            energy_budget_delta: 0.0,
            reason: "",
        }
    }
}
