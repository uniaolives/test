pub use crate::cge_constitution::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherenceDimension {
    Allocation, // C1
    Stability,  // C2
    Temporality, // C3
    Security,    // C4
    Resilience,  // C6
    Volume,
    Curvature,
    Homology,
}

pub struct CGEState {
    pub phi: f64,
}

pub struct CGEViolation {
    pub message: String,
}
