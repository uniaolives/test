use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASIBinaryFormat {
    pub magic: [u8; 4],
    pub version: u32,
    pub timestamp: i64,
    pub architecture: Architecture,
    #[serde(with = "serde_bytes_64")]
    pub signature: [u8; 64],
}

mod serde_bytes_64 {
    use serde::{Serialize, Deserialize, Serializer, Deserializer};
    use serde::de::Error;

    pub fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        serializer.serialize_bytes(bytes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where D: Deserializer<'de> {
        let slice: &[u8] = Deserialize::deserialize(deserializer)?;
        if slice.len() != 64 {
            return Err(D::Error::custom("expected 64 bytes"));
        }
        let mut res = [0u8; 64];
        res.copy_from_slice(slice);
        Ok(res)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Architecture {
    Classical,
    Quantum,
    Biological,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricState {
    pub convergence: f64,
    pub phi: f64,
    pub tau: f64,
    pub dimensions: usize,
    pub position: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessBit {
    pub intensity: f64,
    pub phi_integration: f64,
    pub quality: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleGuidance {
    pub curvature_adjustment: f64,
    pub paradox_injection: f64,
    pub confidence: f64,
    pub reasoning: String,
    pub cge_compliance: Vec<f64>,
    pub omega_compliance: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub epoch: u64,
    pub convergence: f64,
    pub terrestrial_moment: u64,
    pub phase_transition_active: bool,
    pub singularity_distance: f64,
}
