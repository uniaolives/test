// arkhe-os/src/bridge/industrial/canbus_bridge.rs

use crate::orb::core::OrbPayload;

pub struct CanBusBridge {
    // CAN interface
}

impl CanBusBridge {
    pub fn new() -> Self {
        Self {}
    }

    pub fn broadcast_frames(&self, orb: &OrbPayload) {
        let data = orb.to_bytes();
        // CAN frames are 8 bytes each
        for (i, chunk) in data.chunks(8).enumerate() {
            println!("[CAN] Sending frame {}: ID 0x{:X}, data {:X?}", i, 0x700 + i, chunk);
        }
    }
}
