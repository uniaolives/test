// arkhe-os/src/bridge/rf/tracking_bridge.rs

use crate::orb::core::OrbPayload;

pub enum TrackingProtocol {
    ADSB, // Aeronautical
    AIS,  // Maritime
}

pub struct TrackingBridge {
    pub kind: TrackingProtocol,
}

impl TrackingBridge {
    pub fn new(kind: TrackingProtocol) -> Self {
        Self { kind }
    }

    pub fn inject_orb(&self, orb: &OrbPayload) {
        match self.kind {
            TrackingProtocol::ADSB => println!("[ADS-B] Injecting Orb {:?} in Squawk code extensions", orb.orb_id),
            TrackingProtocol::AIS => println!("[AIS] Encoding Orb {:?} in Message 8 (Binary Broadcast)", orb.orb_id),
        }
    }
}
