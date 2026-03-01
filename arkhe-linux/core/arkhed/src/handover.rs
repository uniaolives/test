use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;

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
}
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use uuid::Uuid;
use bincode;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[repr(C)]
pub struct HandoverHeader {
    pub magic: [u8; 4],           // "ARKH"
    pub version: u8,              // 0x01
    pub handover_type: u8,        // 0x01: excitatory, 0x02: inhibitory, 0x03: meta, 0x04: structural
    pub flags: u16,
    pub id: Uuid,
    pub emitter_id: u64,
    pub receiver_id: u64,
    pub timestamp_physical: u64,
    pub timestamp_logical: u32,
    pub entropy_cost: f32,
    pub half_life: f32,           // Ω+205.1: life expectancy in ms
    pub payload_length: u32,
    pub reserved: u32,            // Alignment and expansion
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Handover {
    pub header: HandoverHeader,
    pub payload: Vec<u8>,
    #[serde(with = "serde_signature")]
    pub signature: [u8; 2427],    // Placeholder for Dilithium5 signature
}

mod serde_signature {
    use super::*;
    use serde::ser::SerializeTuple;
    use serde::de::{Visitor, SeqAccess, Error};
    use std::fmt;

    pub fn serialize<S>(sig: &[u8; 2427], serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let mut tup = serializer.serialize_tuple(2427)?;
        for byte in sig.iter() {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 2427], D::Error>
    where D: Deserializer<'de> {
        struct SignatureVisitor;
        impl<'de> Visitor<'de> for SignatureVisitor {
            type Value = [u8; 2427];
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a byte array of length 2427")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where A: SeqAccess<'de> {
                let mut sig = [0u8; 2427];
                for i in 0..2427 {
                    sig[i] = seq.next_element()?.ok_or_else(|| Error::invalid_length(i, &self))?;
                }
                Ok(sig)
            }
        }
        deserializer.deserialize_tuple(2427, SignatureVisitor)
    }
}

impl Handover {
    pub fn new(
        handover_type: u8,
        emitter: u64,
        receiver: u64,
        entropy_cost: f32,
        half_life: f32,
        payload: Vec<u8>,
    ) -> Self {
        let payload_length = payload.len() as u32;
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
            payload_length,
            reserved: 0,
        };

        Self {
            header,
            payload,
            signature: [0u8; 2427],
        }
    }

    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

pub struct HandoverManager;

impl HandoverManager {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            history: Mutex::new(Vec::new()),
        }
    }

    pub async fn process_queue(&self) -> Result<(), anyhow::Error> {
        let mut queue = self.queue.lock().unwrap();
        let mut history = self.history.lock().unwrap();
        while let Some(packet) = queue.pop_front() {
            tracing::info!("Processing handover: {}", packet.id);
            history.push(packet);
        }
        Ok(())
    }

    pub fn enqueue(&self, packet: HandoverPacket) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(packet);
    }

    pub async fn broadcast_phi(&self, _phi: f64) {
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
        let payload = b"Hello, Arkhe!".to_vec();
        let handover = Handover::new(0x01, 1, 2, 0.5, 100.0, payload.clone());

        let serialized = handover.serialize().expect("Failed to serialize handover");
        let deserialized = Handover::deserialize(&serialized).expect("Failed to deserialize handover");

        assert_eq!(handover.header.magic, deserialized.header.magic);
        assert_eq!(handover.header.id, deserialized.header.id);
        assert_eq!(handover.payload, deserialized.payload);
        assert_eq!(handover.signature, deserialized.signature);
    }
}
