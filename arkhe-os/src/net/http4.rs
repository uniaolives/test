// arkhe-os/src/net/http4.rs

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::security::grail::GrailProof;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Http4Method {
    OBSERVE,
    EMIT,
    ENTANGLE,
    COLLAPSE,
    QUANTIZE,
    PROPAGATE,
    RECALL,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConfinementMode {
    INFINITE_WELL,
    FINITE_WELL,
    BARRIER,
    FREE,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParadoxPolicy {
    REJECT,
    MERGE,
    COLLAPSE,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalHeaders {
    pub origin: DateTime<Utc>,
    pub target: DateTime<Utc>,
    pub lambda_2: f64,
    pub confinement: ConfinementMode,
    pub paradox_policy: ParadoxPolicy,
    pub mobius_twist: f64,
    pub oam_state: Option<i32>,            // Orbital Angular Momentum (l)
    pub retrocausal_timestamp: Option<i64>, // CSU-adjusted timestamp
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Http4Request {
    pub method: Http4Method,
    pub uqi: String,
    pub headers: TemporalHeaders,
    #[serde(default)]
    pub payload: Vec<u8>,
    #[serde(default)]
    pub grail_signature: Option<GrailProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Http4Response {
    State(Vec<u8>),
    ChannelOpen,
    RealityPreserved,
    Error(String),
}
