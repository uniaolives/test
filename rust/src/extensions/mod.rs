// rust/src/extensions/mod.rs
pub mod biological;
pub mod planetary;
pub mod quantum_gravity;
pub mod agi_geometric;
pub mod asi_structured;

pub struct ExtensionReport {
    pub scale: String,
    pub status: String,
    pub coherence: f64,
}
