pub mod constitutional;

pub enum ObjectiveResult {
    Safe,
    Violation(String),
    Unknown,
}

pub trait Objective {
    fn evaluate(&self) -> ObjectiveResult;
}

#[cfg(feature = "z3-enabled")]
pub use constitutional::SovereigntyObjective;
