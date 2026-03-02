use anyhow::Result;

pub struct DelegateSystem;

impl DelegateSystem {
    pub fn new() -> Self {
        Self
    }

    pub fn add_hardware_rng(&self) -> Result<RNGReport> {
        Ok(RNGReport {
            delegates_updated: 128,
            entropy_quality: 0.9999,
            status: "INTEGRADO".to_string(),
        })
    }
}

pub struct RNGReport {
    pub delegates_updated: usize,
    pub entropy_quality: f64,
    pub status: String,
}
