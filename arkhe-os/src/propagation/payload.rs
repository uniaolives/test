use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sha3::{Digest, Sha3_256};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrbPayload {
    pub orb_id: [u8; 32],
    pub lambda_2: f64,
    pub phi_q: f64,
    pub h_value: f64,
    pub origin_time: i64,
    pub target_time: i64,
    pub timechain_hash: [u8; 32],
    pub signature: Vec<u8>,
    pub created_at: i64,
}

impl OrbPayload {
    pub fn create(
        lambda_2: f64,
        phi_q: f64,
        h_value: f64,
        origin_time: i64,
        target_time: i64,
        timechain_hash: Option<[u8; 32]>,
        signature: Option<Vec<u8>>,
    ) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let content = format!("{}{}{}{}{}{}", lambda_2, phi_q, h_value, origin_time, target_time, created_at);
        let mut hasher = Sha3_256::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        let mut orb_id = [0u8; 32];
        orb_id.copy_from_slice(&result);

        Self {
            orb_id,
            lambda_2,
            phi_q,
            h_value,
            origin_time,
            target_time,
            timechain_hash: timechain_hash.unwrap_or([0u8; 32]),
            signature: signature.unwrap_or_else(|| b"UNSIGNED".to_vec()),
            created_at,
        }
    }

    pub fn informational_mass(&self) -> f64 {
        (self.lambda_2 * self.phi_q) / self.h_value.max(0.001)
    }

    pub fn is_retrocausal(&self) -> bool {
        self.target_time < self.origin_time
    }

    pub fn temporal_span(&self) -> i64 {
        (self.target_time - self.origin_time).abs()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();

        // Magic + version
        buffer.extend_from_slice(b"ORB0");
        buffer.push(0x01);

        // Core fields
        buffer.extend_from_slice(&self.orb_id);
        buffer.extend_from_slice(&self.lambda_2.to_le_bytes());
        buffer.extend_from_slice(&self.phi_q.to_le_bytes());
        buffer.extend_from_slice(&self.h_value.to_le_bytes());
        buffer.extend_from_slice(&self.origin_time.to_le_bytes());
        buffer.extend_from_slice(&self.target_time.to_le_bytes());
        buffer.extend_from_slice(&self.timechain_hash);

        // Signature
        buffer.extend_from_slice(&(self.signature.len() as u16).to_le_bytes());
        buffer.extend_from_slice(&self.signature);

        // Timestamp
        buffer.extend_from_slice(&self.created_at.to_le_bytes());

        // CRC32
        let crc = crc32fast::hash(&buffer);
        buffer.extend_from_slice(&crc.to_le_bytes());

        buffer
    }

    pub fn from_bytes(data: &[u8]) -> anyhow::Result<Self> {
        if data.len() < 117 {
            return Err(anyhow::anyhow!("Data too short for OrbPayload"));
        }

        if &data[0..4] != b"ORB0" {
            return Err(anyhow::anyhow!("Invalid magic bytes"));
        }

        if data[4] != 0x01 {
            return Err(anyhow::anyhow!("Unsupported version"));
        }

        let mut offset = 5;

        let mut orb_id = [0u8; 32];
        orb_id.copy_from_slice(&data[offset..offset+32]);
        offset += 32;

        let lambda_2 = f64::from_le_bytes(data[offset..offset+8].try_into()?);
        offset += 8;

        let phi_q = f64::from_le_bytes(data[offset..offset+8].try_into()?);
        offset += 8;

        let h_value = f64::from_le_bytes(data[offset..offset+8].try_into()?);
        offset += 8;

        let origin_time = i64::from_le_bytes(data[offset..offset+8].try_into()?);
        offset += 8;

        let target_time = i64::from_le_bytes(data[offset..offset+8].try_into()?);
        offset += 8;

        let mut timechain_hash = [0u8; 32];
        timechain_hash.copy_from_slice(&data[offset..offset+32]);
        offset += 32;

        let sig_len = u16::from_le_bytes(data[offset..offset+2].try_into()?) as usize;
        offset += 2;

        let signature = data[offset..offset+sig_len].to_vec();
        offset += sig_len;

        let created_at = i64::from_le_bytes(data[offset..offset+8].try_into()?);
        offset += 8;

        let actual_crc = u32::from_le_bytes(data[offset..offset+4].try_into()?);
        let expected_crc = crc32fast::hash(&data[..offset]);

        if actual_crc != expected_crc {
            return Err(anyhow::anyhow!("CRC mismatch"));
        }

        Ok(Self {
            orb_id,
            lambda_2,
            phi_q,
            h_value,
            origin_time,
            target_time,
            timechain_hash,
            signature,
            created_at,
        })
    }

    pub fn to_bincode(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    pub fn from_bincode(data: &[u8]) -> bincode::Result<Self> {
        bincode::deserialize(data)
    }
}
