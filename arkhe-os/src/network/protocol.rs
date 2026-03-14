// arkhe-os/src/network/protocol.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Http4Method {
    EMIT,
    OBSERVE,
    ENTANGLE,
    COLLAPSE,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TemporalHeaders {
    pub x_temporal_origin: String,
    pub x_temporal_target: String,
    pub x_lambda_2: f64,
    pub x_oam_index: u8,
}

pub struct OrbRequest {
    pub method: Http4Method,
    pub headers: TemporalHeaders,
    pub payload: Vec<u8>,
}

pub struct OrbResponse {
    pub status_code: u16,
    pub status_message: String,
    pub data: Option<Vec<u8>>,
}

impl OrbResponse {
    pub fn accepted() -> Self {
        Self {
            status_code: 202,
            status_message: "Transmission Initiated".to_string(),
            data: None,
        }
    }

    pub fn insufficient_coherence() -> Self {
        Self {
            status_code: 507,
            status_message: "Insufficient Coherence".to_string(),
            data: None,
        }
    }
}
