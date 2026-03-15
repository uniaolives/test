use tokio_modbus::prelude::*;
use crate::Result;

pub struct ModbusController {
    pub address: String,
}

impl ModbusController {
    pub async fn connect(addr: &str) -> Result<Self> {
        println!("[OrbVM] Connecting to Modbus at {}...", addr);
        Ok(Self { address: addr.to_string() })
    }
}
