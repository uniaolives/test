// arkhe-os/src/bridge/tcpip/gopher_bridge.rs

use crate::orb::core::OrbPayload;
use crate::bridge::BridgeError;
use std::io::Write;

pub struct GopherBridge {
    pub server: String,
}

impl GopherBridge {
    pub fn new(server: &str) -> Self {
        Self { server: server.to_string() }
    }

    pub async fn transmit(&self, orb: &OrbPayload) -> Result<(), BridgeError> {
        let data = orb.to_bytes();
        println!("[Gopher] Sending Orb {:?} as a binary resource to {}", orb.orb_id, self.server);
        // Gopher is just TCP. Send selector then data.
        Ok(())
    }
}
