use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlertPayload {
    pub alert_id: String,
    pub node: String,
    pub probability: f64,
    pub severity: String,
    pub timestamp: i64,
    pub actions_taken: Vec<String>,
}

impl crate::handover::Handover {
    pub fn new_alert(payload: AlertPayload, sk: &pqcrypto_dilithium::dilithium5::SecretKey) -> Self {
        let json_payload = serde_json::to_vec(&payload).unwrap();
        Self::new(0x05, 0, 0, 0.0, 0.0, json_payload, sk)
    }
}
