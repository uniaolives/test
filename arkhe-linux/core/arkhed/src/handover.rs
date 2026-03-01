use serde::{Serialize, Deserialize};
use uuid::Uuid;
use pqcrypto_dilithium::dilithium5::*;
use pqcrypto_traits::sign::{PublicKey as _, SecretKey as _, DetachedSignature as _};
use byteorder::{ByteOrder, LittleEndian};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct HandoverHeader {
    pub magic: [u8; 4],            // "ARKH"
    pub version: u8,               // 0x01
    pub handover_type: HandoverType,
    pub flags: u16,
    pub id: [u8; 16],              // UUID binário
    pub emitter_id: [u8; 8],
    pub receiver_id: [u8; 8],
    pub timestamp_physical: i64,   // Nanoseconds since epoch
    pub timestamp_logical: u32,    // Hybrid Logical Clock counter
    pub entropy_cost: f32,         // AEU cost
    pub half_life_ms: f32,         // Average lifetime (ms)
    pub payload_length: u32,
}

pub struct Handover {
    pub header: HandoverHeader,
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,        // Dilithium5 signature (2427 bytes)
}

impl Handover {
    pub fn new(
        handover_type: HandoverType,
        emitter_id: [u8; 8],
        receiver_id: [u8; 8],
        payload: Vec<u8>,
        entropy_cost: ArkheEntropyUnit,
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

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64 + self.payload.len() + self.signature.len());

        buf.extend_from_slice(&self.header.magic);
        buf.push(self.header.version);
        buf.push(self.header.handover_type as u8);

        let mut flags_bytes = [0u8; 2];
        LittleEndian::write_u16(&mut flags_bytes, self.header.flags);
        buf.extend_from_slice(&flags_bytes);

        buf.extend_from_slice(&self.header.id);
        buf.extend_from_slice(&self.header.emitter_id);
        buf.extend_from_slice(&self.header.receiver_id);

        let mut ts_phys_bytes = [0u8; 8];
        LittleEndian::write_i64(&mut ts_phys_bytes, self.header.timestamp_physical);
        buf.extend_from_slice(&ts_phys_bytes);

        let mut ts_log_bytes = [0u8; 4];
        LittleEndian::write_u32(&mut ts_log_bytes, self.header.timestamp_logical);
        buf.extend_from_slice(&ts_log_bytes);

        let mut entropy_bytes = [0u8; 4];
        LittleEndian::write_f32(&mut entropy_bytes, self.header.entropy_cost);
        buf.extend_from_slice(&entropy_bytes);

        let mut half_life_bytes = [0u8; 4];
        LittleEndian::write_f32(&mut half_life_bytes, self.header.half_life_ms);
        buf.extend_from_slice(&half_life_bytes);

        let mut plen_bytes = [0u8; 4];
        LittleEndian::write_u32(&mut plen_bytes, self.header.payload_length);
        buf.extend_from_slice(&plen_bytes);

        buf.extend_from_slice(&self.payload);
        buf.extend_from_slice(&self.signature);
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 64 {
            return Err("buffer too small for header");
        }

        let magic = &bytes[0..4];
        if magic != b"ARKH" {
            return Err("invalid magic");
        }

        let version = bytes[4];
        let handover_type = match bytes[5] {
            0x01 => HandoverType::Excitatory,
            0x02 => HandoverType::Inhibitory,
            0x03 => HandoverType::Meta,
            0x04 => HandoverType::Structural,
            _ => return Err("invalid handover type"),
        };

        let flags = LittleEndian::read_u16(&bytes[6..8]);
        let mut id = [0u8; 16];
        id.copy_from_slice(&bytes[8..24]);
        let mut emitter_id = [0u8; 8];
        emitter_id.copy_from_slice(&bytes[24..32]);
        let mut receiver_id = [0u8; 8];
        receiver_id.copy_from_slice(&bytes[32..40]);

        let timestamp_physical = LittleEndian::read_i64(&bytes[40..48]);
        let timestamp_logical = LittleEndian::read_u32(&bytes[48..52]);
        let entropy_cost = LittleEndian::read_f32(&bytes[52..56]);
        let half_life_ms = LittleEndian::read_f32(&bytes[56..60]);
        let payload_length = LittleEndian::read_u32(&bytes[60..64]) as usize;

        if bytes.len() < 64 + payload_length {
            return Err("buffer too small for payload");
        }

        let payload = bytes[64..64 + payload_length].to_vec();
        let signature = bytes[64 + payload_length..].to_vec();

        let header = HandoverHeader {
            magic: [magic[0], magic[1], magic[2], magic[3]],
            version,
            handover_type,
            flags,
            id,
            emitter_id,
            receiver_id,
            timestamp_physical,
            timestamp_logical,
            entropy_cost,
            half_life_ms,
            payload_length: payload_length as u32,
        };

        Ok(Self {
            header,
            payload,
            signature,
        })
    }
}

pub struct HandoverManager {
    queue: Vec<Handover>,
}

impl HandoverManager {
    pub fn new() -> Self {
        Self {
            queue: Vec::new(),
        }
    }

    pub async fn process_queue(&mut self) -> Result<(), anyhow::Error> {
        for packet in self.queue.drain(..) {
            tracing::info!("Processing handover from {:?} to {:?}",
                packet.header.emitter_id, packet.header.receiver_id);
            // Validation logic will go here
        }
        Ok(())
    }

    pub async fn broadcast_phi(&self, _phi: f64) {
        // Implementation for phi broadcasting
    }

    pub async fn send_system_notification(&self, msg: &str) -> Result<(), anyhow::Error> {
        tracing::info!("System notification: {}", msg);
        Ok(())
    }

    pub fn queue_handover(&mut self, packet: Handover) {
        self.queue.push(packet);
    }
}
