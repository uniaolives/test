use num_complex::Complex;
use uuid::Uuid;
use crate::propagation::payload::OrbPayload;

/// An impossible timeline where physical laws diverge
#[derive(Debug, Clone)]
pub struct ImpossibleTimeline {
    pub timeline_id: Uuid,
    pub constants: DivergentConstants,
    pub impossibility_class: ImpossibilityClass,
    pub lambda_2: f64,
    pub tunnel_probability: f64,
}

#[derive(Debug, Clone)]
pub struct DivergentConstants {
    /// Speed of light (can be ∞ or complex)
    pub c: Complex<f64>,
    /// Planck's constant (can be 0)
    pub h_bar: f64,
    /// Gravitational constant (can be negative)
    pub g: f64,
    /// Cosmological constant
    pub lambda: f64,
    /// Maximum coherence (can be > 1)
    pub lambda_max: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpossibilityClass {
    /// Tachyons are normal particles
    TachyonicNorm,
    /// Entropy flows backwards
    ReverseEntropy,
    /// Time is a spatial dimension
    TimeSpatialized,
    /// Superposition is macroscopic
    MacroscopicQuantum,
    /// Information is not conserved
    NonConservedInfo,
    /// Constants vary across spacetime
    VariableConstants,
    /// Universe is an idealistic ontology
    IdealisticOntology,
}

impl ImpossibleTimeline {
    /// Calculates the "distance" from a standard timeline
    pub fn divergence_metric(&self) -> f64 {
        let c_std = 299792458.0;
        let h_std = 1.054e-34;
        let g_std = 6.674e-11;

        let c_div = (self.constants.c.norm() - c_std).abs() / c_std;
        let h_div = (self.constants.h_bar - h_std).abs() / h_std;
        let g_div = (self.constants.g - g_std).abs() / g_std;

        (c_div + h_div + g_div) / 3.0
    }

    pub fn can_support_orb(&self, orb: &OrbPayload) -> bool {
        match self.impossibility_class {
            ImpossibilityClass::TachyonicNorm => true,
            ImpossibilityClass::ReverseEntropy => orb.lambda_2 > 0.8,
            ImpossibilityClass::MacroscopicQuantum => true,
            _ => self.tunnel_probability > 0.01,
        }
    }

    pub fn transform_orb(&self, orb: &OrbPayload) -> TransformedOrb {
        let mut transformed = orb.clone();

        match self.impossibility_class {
            ImpossibilityClass::TachyonicNorm => {
                transformed.phi_q *= -1.0;
            }
            ImpossibilityClass::ReverseEntropy => {
                transformed.h_value = -transformed.h_value;
            }
            ImpossibilityClass::MacroscopicQuantum => {
                transformed.lambda_2 = 1.0;
            }
            _ => {}
        }

        TransformedOrb {
            original: orb.clone(),
            transformed,
            timeline: self.timeline_id,
        }
    }
}

pub struct TransformedOrb {
    pub original: OrbPayload,
    pub transformed: OrbPayload,
    pub timeline: Uuid,
}
