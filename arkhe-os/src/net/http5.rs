// arkhe-os/src/net/http5.rs

use serde::{Deserialize, Serialize};
use crate::net::http4::{Http4Method, TemporalHeaders, Http4Response};
use crate::security::grail::GrailProof;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterstellarHeaders {
    pub temporal_headers: TemporalHeaders,
    pub carrier_frequency: f64, // MHz
    pub oam_topology_l: i32,    // Orbital Angular Momentum
    pub grail_signature: String, // e.g., 6EQUJ5-ALPHA-OMEGA
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Http5Request {
    pub method: Http4Method,
    pub resource: String, // e.g., /heliosphere/voyager_genesis
    pub interstellar_headers: InterstellarHeaders,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Http5Response {
    TimelineStabilized(String),
    InterstellarAnchorLocked,
    Error(String),
}
