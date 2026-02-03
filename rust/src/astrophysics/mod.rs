// rust/src/astrophysics/mod.rs
pub mod n_body;
pub mod kardashev;

pub use n_body::{NBodySimulator, CelestialBody, TrajectoryStatus};
