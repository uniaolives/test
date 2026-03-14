//! Network layer - SSH, Tor, Modbus

pub mod ssh;
pub mod tor;
pub mod modbus;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub enable_ssh: bool,
    pub ssh_port: u16,
    pub enable_tor: bool,
    pub tor_socks_port: u16,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            enable_ssh: true,
            ssh_port: 3141,
            enable_tor: true,
            tor_socks_port: 9050,
        }
    }
}
