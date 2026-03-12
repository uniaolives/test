// arkhe-os/src/bridge/industrial/modbus_bridge.rs

use tokio_modbus::prelude::*;
use crate::orb::core::OrbPayload;
use crate::bridge::BridgeError;
use tokio_modbus::client::Context;

pub struct ModbusBridge {
    client: Context,
}

impl ModbusBridge {
    pub fn new(ctx: Context) -> Self {
        Self { client: ctx }
    }

    /// Codifica Orb em registradores Modbus
    pub async fn write_orb(&mut self, orb: &OrbPayload, start_register: u16) -> Result<(), BridgeError> {
        let data = orb.to_bytes();

        // Converter bytes para registradores de 16 bits
        let registers: Vec<u16> = data
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    u16::from_be_bytes([chunk[0], chunk[1]])
                } else {
                    u16::from_be_bytes([chunk[0], 0])
                }
            })
            .collect();

        // Escrever em holding registers
        self.client.write_multiple_registers(start_register, &registers).await
            .map_err(|e| BridgeError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?
            .map_err(|e| BridgeError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Modbus exception: {:?}", e))))?;

        Ok(())
    }
}
