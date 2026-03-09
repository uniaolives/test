// arkhe-os/src/bridge/rf/satellite_bridge.rs

use crate::orb::core::OrbPayload;
use super::{RfFrame, ModulationType};
use crc::Crc;

/// Codifica Orb para transmissão via satélite
pub struct SatelliteBridge {
    frequency_hz: u64,
    modulation: ModulationType,
    _uplink_power_watts: f64,
}

impl SatelliteBridge {
    pub fn new(freq: u64, mod_type: ModulationType, power: f64) -> Self {
        Self {
            frequency_hz: freq,
            modulation: mod_type,
            _uplink_power_watts: power,
        }
    }

    /// Codifica Orb em frames de satélite
    pub fn encode_for_satellite(&self, orb: &OrbPayload) -> Vec<RfFrame> {
        let data = orb.to_bytes();
        let mut frames = Vec::new();

        // Tamanho máximo de frame (depende do satélite)
        let frame_size = 1024;
        let crc_algo = Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);

        for chunk in data.chunks(frame_size) {
            // Adicionar prefixo de sincronização
            let mut frame_data = vec![0xAA, 0x55, 0xAA, 0x55]; // Preambule

            // Adicionar header Orb
            frame_data.extend_from_slice(&orb.orb_id);
            frame_data.extend_from_slice(&(chunk.len() as u16).to_be_bytes());

            // Adicionar payload
            frame_data.extend_from_slice(chunk);

            // Adicionar CRC
            let crc_val = crc_algo.checksum(&frame_data);
            frame_data.extend_from_slice(&crc_val.to_be_bytes());

            frames.push(RfFrame {
                data: frame_data,
                frequency_hz: self.frequency_hz,
                modulation: self.modulation.clone(),
            });
        }

        frames
    }
}
