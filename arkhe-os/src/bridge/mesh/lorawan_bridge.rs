// arkhe-os/src/bridge/mesh/lorawan_bridge.rs

use crate::orb::core::OrbPayload;
use crc::Crc;

pub struct LoRaWanBridge {
    _dev_eui: [u8; 8],
    _app_eui: [u8; 8],
    _app_key: [u8; 16],
}

impl LoRaWanBridge {
    pub fn new(dev_eui: [u8; 8], app_eui: [u8; 8], app_key: [u8; 16]) -> Self {
        Self { _dev_eui: dev_eui, _app_eui: app_eui, _app_key: app_key }
    }

    /// Codifica Orb para payload LoRaWAN (máx 51 bytes)
    pub fn encode(&self, orb: &OrbPayload) -> Vec<u8> {
        let mut payload = Vec::with_capacity(51);
        let crc_algo = Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);

        // Magic bytes
        payload.extend_from_slice(b"OR");

        // Orb ID truncado (8 bytes)
        payload.extend_from_slice(&orb.orb_id[..8]);

        // Lambda_2 (2 bytes, fixed point)
        payload.extend_from_slice(&(orb.lambda_2 as u16).to_be_bytes());

        // Phi_q (1 byte, compressed)
        payload.push((orb.phi_q * 10.0) as u8);

        // Timestamp delta (2 bytes)
        let delta = (orb.origin_time - 1700000000) as u16;
        payload.extend_from_slice(&delta.to_be_bytes());

        // CRC (2 bytes)
        let crc_val = crc_algo.checksum(&payload);
        payload.extend_from_slice(&crc_val.to_be_bytes());

        payload
    }
}
