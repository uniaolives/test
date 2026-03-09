// arkhe-os/src/bridge/mesh/ble_bridge.rs

use crate::orb::core::OrbPayload;
use uuid::Uuid;

pub struct BleBridge {
    _service_uuid: Uuid,
    _characteristic_uuid: Uuid,
}

impl BleBridge {
    pub fn new(service: Uuid, charac: Uuid) -> Self {
        Self { _service_uuid: service, _characteristic_uuid: charac }
    }

    /// Divide Orb em chunks para BLE (máx 20 bytes por pacote)
    pub fn chunk(&self, orb: &OrbPayload) -> Vec<Vec<u8>> {
        let data = orb.to_bytes();

        data.chunks(18)
            .map(|chunk| {
                let mut packet = Vec::with_capacity(20);

                // Header (2 bytes)
                packet.push(0xAB); // Orb marker
                packet.push(chunk.len() as u8);

                // Payload (18 bytes max)
                packet.extend_from_slice(chunk);

                packet
            })
            .collect()
    }
}
