use arkhe_manifold::{QuantumState};
use crate::types;

pub struct ObjectQuantumMapper;

impl ObjectQuantumMapper {
    pub fn map(obj: &types::FoundryObject) -> QuantumState {
        let _phi = obj.properties.get("phi").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let dim = 2;
        QuantumState::maximally_mixed(dim)
    }
}
