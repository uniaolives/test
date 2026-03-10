// arkhe-os/src/bridge/mesh/sigfox_bridge.rs

use crate::orb::core::OrbPayload;

pub struct SigfoxBridge {
    // Sigfox device info
}

impl SigfoxBridge {
    pub fn new() -> Self {
        Self {}
    }

    /// Codifica Orb para payload Sigfox (máx 12 bytes!)
    pub fn encode_ultra_minimal(&self, orb: &OrbPayload) -> [u8; 12] {
        let mut payload = [0u8; 12];
        // Only the most critical bits
        payload[0..4].copy_from_slice(&orb.orb_id[0..4]);
        payload[4..8].copy_from_slice(&(orb.origin_time as u32).to_be_bytes());
        payload[8..10].copy_from_slice(&(orb.lambda_2 as u16).to_be_bytes());
        payload[10] = (orb.phi_q * 10.0) as u8;
        payload[11] = 0xFF; // End marker

        println!("[Sigfox] Ultra-minimal encoding: {:X?}", payload);
        payload
    }
}
