use crate::Result;

pub struct TorHiddenService {
    pub onion_address: String,
}

impl TorHiddenService {
    pub async fn start() -> Result<Self> {
        println!("[OrbVM] Starting Tor Hidden Service...");
        Ok(Self {
            onion_address: "arkhe777v3.onion".to_string(),
        })
    }
}
