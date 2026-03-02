pub mod asi_core;
pub mod manifold;
pub mod z3_verifier;
pub mod thermodynamics;
pub mod constitution;
pub mod self_modification;
pub mod verification;
pub mod depin;

use tokio::sync::Mutex;
use std::sync::Arc;
use ndarray::Array2;
use num_complex::Complex64;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use pqcrypto_dilithium::dilithium5::*;
use pqcrypto_traits::sign::{PublicKey as _, SecretKey as _, DetachedSignature as _};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum HandoverType {
    Excitatory = 0x01,
    Inhibitory = 0x02,
    Meta = 0x03,
    Structural = 0x04,
    RLTransition = 0x06,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[repr(C)]
pub struct HandoverHeader {
    pub magic: [u8; 4],
    pub version: u8,
    pub handover_type: HandoverType,
    pub flags: u16,
    pub id: Uuid,
    pub emitter_id: u64,
    pub receiver_id: u64,
    pub timestamp_physical: u64,
    pub timestamp_logical: u32,
    pub entropy_cost: f32,
    pub half_life: f32,
    pub payload_length: u32,
    pub reserved: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Handover {
    pub header: HandoverHeader,
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,
}

impl Handover {
    pub fn new(
        handover_type: HandoverType,
        emitter: u64,
        receiver: u64,
        entropy_cost: f32,
        half_life: f32,
        payload: Vec<u8>,
    ) -> Self {
        let header = HandoverHeader {
            magic: *b"ARKH",
            version: 0x01,
            handover_type,
            flags: 0,
            id: Uuid::new_v4(),
            emitter_id: emitter,
            receiver_id: receiver,
            timestamp_physical: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            timestamp_logical: 0,
            entropy_cost,
            half_life,
            payload_length: payload.len() as u32,
            reserved: 0,
        };

        Self {
            header,
            payload,
            signature: Vec::new(),
        }
    }

    pub fn sign(&mut self, secret_key_bytes: &[u8]) -> Result<(), String> {
        let mut data = bincode::serialize(&self.header).map_err(|e| e.to_string())?;
        data.extend_from_slice(&self.payload);

        let sk = SecretKey::from_bytes(secret_key_bytes).map_err(|e| e.to_string())?;
        let sig = detached_sign(&data, &sk);
        self.signature = sig.as_bytes().to_vec();
        Ok(())
    }

    pub fn verify(&self, public_key_bytes: &[u8]) -> bool {
        let mut data = match bincode::serialize(&self.header) {
            Ok(d) => d,
            Err(_) => return false,
        };
        data.extend_from_slice(&self.payload);

        let sig = match DetachedSignature::from_bytes(&self.signature) {
            Ok(s) => s,
            Err(_) => return false,
        };

        let pk = match PublicKey::from_bytes(public_key_bytes) {
            Ok(p) => p,
            Err(_) => return false,
        };

        verify_detached_signature(&sig, &data, &pk).is_ok()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RLTransition {
    pub state: Vec<f64>,
    pub action: u32,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
    pub timestamp: i64,
    pub node_id: String,
    pub model_version: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelVersion {
    pub version: String,
    pub mlflow_run_id: String,
    pub artifact_uri: String,
    pub metrics: std::collections::HashMap<String, f64>,
    pub parameters: std::collections::HashMap<String, String>,
    pub parent_version: Option<String>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub density_matrix: Array2<Complex64>,
}

impl QuantumState {
    pub fn new(dim: usize) -> Self {
        let mut dm = Array2::from_elem((dim, dim), Complex64::new(0.0, 0.0));
        dm[[0, 0]] = Complex64::new(1.0, 0.0);
        Self { density_matrix: dm }
    }

    pub fn maximally_mixed(dim: usize) -> Self {
        let dm = Array2::from_elem((dim, dim), Complex64::new(1.0 / dim as f64, 0.0));
        Self { density_matrix: dm }
    }

    pub fn dim(&self) -> usize {
        self.density_matrix.nrows()
    }

    pub fn evolve(&mut self, op: &KrausOperator) {
        let mut new_dm = Array2::from_elem(self.density_matrix.dim(), Complex64::new(0.0, 0.0));
        for k in &op.operators {
            let kt = k.t();
            let term = k.dot(&self.density_matrix).dot(&kt);
            new_dm = new_dm + term;
        }
        self.density_matrix = new_dm;
    }

    pub fn von_neumann_entropy(&self) -> f64 {
        let purity = self.density_matrix.dot(&self.density_matrix).diag().sum().re;
        if purity >= 1.0 { 0.0 } else { (1.0 - purity).abs() }
    }

    pub fn fidelity(&self, other: &Self) -> f64 {
        self.density_matrix.dot(&other.density_matrix).diag().sum().re
    }
}

#[derive(Debug, Clone)]
pub struct KrausOperator {
    pub operators: Vec<Array2<Complex64>>,
}

impl Default for KrausOperator {
    fn default() -> Self {
        // O padrão deve ser a Identidade (No-Op), não o operador nulo
        let dim = 2; // Dimensão padrão
        let mut op = Array2::from_elem((dim, dim), Complex64::new(0.0, 0.0));
        for i in 0..dim {
            op[[i, i]] = Complex64::new(1.0, 0.0);
        }
        Self { operators: vec![op] }
    }
}

impl KrausOperator {
    pub fn from_handover(h: &Handover, dim: usize) -> Self {
        let mut op = Array2::from_elem((dim, dim), Complex64::new(0.0, 0.0));
        let factor = (1.0 - h.header.entropy_cost as f64).max(0.0).sqrt();
        for i in 0..dim {
            op[[i, i]] = Complex64::new(factor, 0.0);
        }
        Self { operators: vec![op] }
    }
}

pub async fn run_engine() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_nanos()
        .init();

    log::info!("🜁 ARKHE(n) QUANTUM OS – PROTOCOLO Ω+210");

    let manifold = Arc::new(Mutex::new(crate::manifold::GlobalManifold::new("localhost").await?));

    asi_core::singularity_engine_loop(manifold).await;
}
