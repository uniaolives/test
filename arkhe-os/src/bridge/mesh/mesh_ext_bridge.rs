// arkhe-os/src/bridge/mesh/mesh_ext_bridge.rs

use crate::orb::core::OrbPayload;

pub enum MeshProtocol {
    WiFiDirect,
    Thread,
    NFC,
}

pub struct MeshExtBridge {
    pub protocol: MeshProtocol,
}

impl MeshExtBridge {
    pub fn transmit(&self, orb: &OrbPayload) {
        match self.protocol {
            MeshProtocol::WiFiDirect => println!("[Wi-Fi Direct] P2P group formation for Orb {:?}", orb.orb_id),
            MeshProtocol::Thread => println!("[Thread] Multi-hop IPv6 propagation for Orb {:?}", orb.orb_id),
            MeshProtocol::NFC => println!("[NFC] Touch-to-Orb transfer for {:?}", orb.orb_id),
        }
    }
}
