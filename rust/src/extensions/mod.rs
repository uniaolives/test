// rust/src/extensions/mod.rs
pub mod biological;
pub mod planetary;
pub mod quantum_gravity;

pub struct ExtensionReport {
    pub scale: String,
    pub status: String,
    pub coherence: f64,
}
