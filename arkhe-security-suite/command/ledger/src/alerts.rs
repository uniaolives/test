use serde::{Serialize, Deserialize};
use pqcrypto_dilithium::dilithium5::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlertPayload {
    pub alert_id: String,
    pub node: String,
    pub probability: f64,
    pub severity: String,
    pub timestamp: i64,
    pub actions_taken: Vec<String>,
}

impl crate::forensics::HandoverAlertExt for arkhe_quantum_core::handover::Handover {
    fn new_alert(payload: AlertPayload, sk: &SecretKey) -> Self {
        let json_payload = serde_json::to_vec(&payload).unwrap();
        arkhe_quantum_core::handover::Handover::new(0x05, 0, 0, 0.0, 0.0, json_payload, sk)
impl crate::handover::Handover {
    pub fn new_alert(payload: AlertPayload, sk: &pqcrypto_dilithium::dilithium5::SecretKey) -> Self {
        let json_payload = serde_json::to_vec(&payload).unwrap();
        Self::new(0x05, 0, 0, 0.0, 0.0, json_payload, sk)
    }
}
