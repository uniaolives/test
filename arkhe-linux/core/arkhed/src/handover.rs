use serde::{Serialize, Deserialize};
use uuid::Uuid;
use pqcrypto_dilithium::dilithium5::*;
use pqcrypto_traits::sign::{PublicKey as _, SecretKey as _, DetachedSignature as _};
use byteorder::{ByteOrder, LittleEndian};
use std::collections::VecDeque;
use std::sync::Mutex;
use crate::entropy::ArkheEntropyUnit;
use crate::hlc::HybridClock;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum HandoverType {
    Excitatory = 0x01,
    Inhibitory = 0x02,
    Meta = 0x03,
    Structural = 0x04,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[repr(C)]
pub struct HandoverHeader {
    pub magic: [u8; 4],            // "ARKH"
    pub version: u8,               // 0x01
    pub handover_type: HandoverType,
    pub flags: u16,
    pub id: [u8; 16],              // UUID binário
    pub emitter_id: u64,
    pub receiver_id: u64,
    pub timestamp_physical: i64,   // Nanoseconds since epoch
    pub timestamp_logical: u32,    // Hybrid Logical Clock counter
    pub entropy_cost: f32,         // AEU cost
    pub half_life_ms: f32,         // Average lifetime (ms)
    pub payload_length: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Handover {
    pub header: HandoverHeader,
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,        // Dilithium5 signature
}

impl Handover {
    pub fn new(
        handover_type: HandoverType,
        emitter_id: u64,
        receiver_id: u64,
        payload: Vec<u8>,
        entropy_cost: f32,
        half_life_ms: f32,
        clock: &mut HybridClock,
        signing_key: &SecretKey,
    ) -> Self {
        let (ts_phys, ts_log) = clock.tick();

        let header = HandoverHeader {
            magic: *b"ARKH",
            version: 0x01,
            handover_type,
            flags: 0,
            id: *Uuid::new_v4().as_bytes(),
            emitter_id,
            receiver_id,
            timestamp_physical: ts_phys,
            timestamp_logical: ts_log,
            entropy_cost,
            half_life_ms,
            payload_length: payload.len() as u32,
        };

        let mut data = bincode::serialize(&header).expect("header serialization failed");
        data.extend_from_slice(&payload);
        let signature = detached_sign(&data, signing_key).as_bytes().to_vec();

        Self {
            header,
            payload,
            signature,
        }
    }

    pub fn verify(&self, public_key: &PublicKey) -> bool {
        let mut data = bincode::serialize(&self.header).expect("header serialization failed");
        data.extend_from_slice(&self.payload);

        let sig = match DetachedSignature::from_bytes(&self.signature) {
            Ok(s) => s,
            Err(_) => return false,
        };

        verify_detached_signature(&sig, &data, public_key).is_ok()
    }

    pub fn survival_probability(&self, transit_time_ms: f32) -> f32 {
        crate::entropy::survival_probability(self.header.half_life_ms, transit_time_ms)
    }

    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HandoverPacket {
    pub id: String,
    pub target: String,
    pub payload: serde_json::Value,
    pub timestamp: u64,
}

pub struct HandoverManager {
    queue: Mutex<VecDeque<HandoverPacket>>,
    history: Mutex<Vec<HandoverPacket>>,
    binary_queue: Mutex<VecDeque<Handover>>,
}

impl HandoverManager {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            history: Mutex::new(Vec::new()),
            binary_queue: Mutex::new(VecDeque::new()),
        }
    }

    pub async fn process_queue(&self) -> Result<(), anyhow::Error> {
        let mut queue = self.queue.lock().unwrap();
        let mut history = self.history.lock().unwrap();
        while let Some(packet) = queue.pop_front() {
            tracing::info!("Processing handover: {}", packet.id);
            history.push(packet);
        }

        let mut b_queue = self.binary_queue.lock().unwrap();
        while let Some(handover) = b_queue.pop_front() {
            tracing::info!("Processing binary handover from {} to {}",
                handover.header.emitter_id, handover.header.receiver_id);
            // Verification and application logic would go here
        }

        Ok(())
    }

    pub fn enqueue(&self, packet: HandoverPacket) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(packet);
    }

    pub fn enqueue_binary(&self, handover: Handover) {
        let mut b_queue = self.binary_queue.lock().unwrap();
        b_queue.push_back(handover);
    }

    pub async fn broadcast_phi(&self, _phi: f64) {
        // Implementation for phi broadcasting
    }

    pub async fn send_system_notification(&self, msg: &str) -> Result<(), anyhow::Error> {
        tracing::info!("System notification: {}", msg);
        Ok(())
    }

    pub fn get_history(&self) -> Vec<HandoverPacket> {
        self.history.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handover_serialization() {
        let (public_key, secret_key) = keypair();
        let mut clock = HybridClock::new();
        let payload = b"Hello, Arkhe!".to_vec();
        let handover = Handover::new(
            HandoverType::Excitatory,
            1,
            2,
            payload.clone(),
            0.5,
            100.0,
            &mut clock,
            &secret_key
        );

        let serialized = handover.serialize().expect("Failed to serialize handover");
        let deserialized = Handover::deserialize(&serialized).expect("Failed to deserialize handover");

        assert_eq!(handover.header.magic, deserialized.header.magic);
        assert_eq!(handover.header.id, deserialized.header.id);
        assert_eq!(handover.payload, deserialized.payload);
        assert_eq!(handover.signature, deserialized.signature);
        assert!(deserialized.verify(&public_key));
    }
}
