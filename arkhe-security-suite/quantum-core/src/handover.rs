use serde::{Serialize, Deserialize};
use uuid::Uuid;
use pqcrypto_dilithium::dilithium5::*;
use pqcrypto_traits::sign::{PublicKey as _, SecretKey as _, DetachedSignature as _};
use byteorder::{ByteOrder, LittleEndian};

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
    pub handover_type: u8,
    pub flags: u16,
    pub id: [u8; 16],
    pub emitter_id: u64,
    pub receiver_id: u64,
    pub timestamp_physical: u64,
    pub timestamp_logical: u32,
    pub entropy_cost: f32,
    pub half_life: f32,
    pub payload_length: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Handover {
    pub header: HandoverHeader,
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,
}

impl Handover {
    pub fn new(
        handover_type: u8,
        emitter_id: u64,
        receiver_id: u64,
        entropy_cost: f32,
        half_life: f32,
        payload: Vec<u8>,
    ) -> Self {
        let header = HandoverHeader {
            magic: *b"ARKH",
            version: 0x01,
            handover_type,
            flags: 0,
            id: *Uuid::new_v4().as_bytes(),
            emitter_id,
            receiver_id,
            timestamp_physical: 0, // Placeholder
            timestamp_logical: 0,
            entropy_cost,
            half_life,
            payload_length: payload.len() as u32,
        };

        Self {
            header,
            payload,
            signature: vec![0u8; 2427], // Placeholder
        }
    }
}
