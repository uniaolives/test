use super::payload::OrbPayload;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct OrbCompressor;

impl OrbCompressor {
    /// Minimal compression for extremely constrained protocols (< 50 bytes).
    /// Format (32 bytes total):
    /// - Orb ID hash (8 bytes, truncated)
    /// - Lambda_2 (2 bytes, scaled)
    /// - Phi_Q (2 bytes, scaled)
    /// - H_value (2 bytes, scaled)
    /// - Origin_time (4 bytes, relative to 2020)
    /// - Target_time (4 bytes, relative to 2020)
    /// - Timechain hash (8 bytes, truncated)
    /// - CRC16 (2 bytes)
    pub fn compress_minimal(orb: &OrbPayload) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(32);

        // Truncated Orb ID
        buffer.extend_from_slice(&orb.orb_id[..8]);

        // Scaled values
        let l2_scaled = (orb.lambda_2 * 65535.0) as u16;
        let pq_scaled = (orb.phi_q * 10430.0) as u16;
        let hv_scaled = (orb.h_value * 65535.0) as u16;

        buffer.extend_from_slice(&l2_scaled.to_le_bytes());
        buffer.extend_from_slice(&pq_scaled.to_le_bytes());
        buffer.extend_from_slice(&hv_scaled.to_le_bytes());

        // Timestamps (relative to 2020-01-01)
        let epoch_2020 = 1577836800i64;
        let origin_rel = (orb.origin_time - epoch_2020) as i32;
        let target_rel = (orb.target_time - epoch_2020) as i32;

        buffer.extend_from_slice(&origin_rel.to_le_bytes());
        buffer.extend_from_slice(&target_rel.to_le_bytes());

        // Truncated Timechain Hash
        buffer.extend_from_slice(&orb.timechain_hash[..8]);

        // CRC16
        let crc = Self::crc16(&buffer);
        buffer.extend_from_slice(&crc.to_le_bytes());

        buffer
    }

    pub fn decompress_minimal(data: &[u8]) -> Result<OrbPayload, String> {
        if data.len() != 32 {
            return Err(format!("Expected 32 bytes, got {}", data.len()));
        }

        let crc_actual = Self::crc16(&data[..30]);
        let crc_expected = u16::from_le_bytes([data[30], data[31]]);
        if crc_actual != crc_expected {
            return Err("CRC mismatch".to_string());
        }

        let mut orb_id = [0u8; 32];
        orb_id[..8].copy_from_slice(&data[0..8]);

        let lambda_2 = u16::from_le_bytes([data[8], data[9]]) as f64 / 65535.0;
        let phi_q = u16::from_le_bytes([data[10], data[11]]) as f64 / 10430.0;
        let h_value = u16::from_le_bytes([data[12], data[13]]) as f64 / 65535.0;

        let epoch_2020 = 1577836800i64;
        let origin_time = i32::from_le_bytes([data[14], data[15], data[16], data[17]]) as i64 + epoch_2020;
        let target_time = i32::from_le_bytes([data[18], data[19], data[20], data[21]]) as i64 + epoch_2020;

        let mut timechain_hash = [0u8; 32];
        timechain_hash[..8].copy_from_slice(&data[22..30]);

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        Ok(OrbPayload {
            orb_id,
            lambda_2,
            phi_q,
            h_value,
            origin_time,
            target_time,
            timechain_hash,
            signature: b"COMPRESSED".to_vec(),
            created_at,
            state_delta: None,
        })
    }

    fn crc16(data: &[u8]) -> u16 {
        let mut crc: u16 = 0xFFFF;
        for &byte in data {
            crc ^= byte as u16;
            for _ in 0..8 {
                if crc & 0x0001 != 0 {
                    crc = (crc >> 1) ^ 0xA001;
                } else {
                    crc >>= 1;
                }
            }
        }
        crc
    }
}
