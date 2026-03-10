// arkhe-os/src/bridge/dark/p2p_dark_bridge.rs

use crate::orb::core::OrbPayload;

pub enum DarkP2PProtocol {
    Freenet,
    Scuttlebutt,
    DAT,
    Hypercore,
}

pub struct DarkP2PBridge {
    pub protocol: DarkP2PProtocol,
}

impl DarkP2PBridge {
    pub async fn transmit(&self, orb: &OrbPayload) {
        match self.protocol {
            DarkP2PProtocol::Freenet => println!("[Freenet] Inserting Orb {:?} in distributed data store", orb.orb_id),
            DarkP2PProtocol::Scuttlebutt => println!("[Scuttlebutt] Appending Orb {:?} to log-chain", orb.orb_id),
            DarkP2PProtocol::DAT => println!("[DAT] Versioned synchronization of Orb {:?}", orb.orb_id),
            DarkP2PProtocol::Hypercore => println!("[Hypercore] Appending Orb {:?} to sparse feed", orb.orb_id),
        }
    }
}
