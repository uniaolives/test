// arkhe-os/src/bridge/tcpip/coap_bridge.rs

use coap::CoAPClient;
use crate::orb::core::OrbPayload;
use crate::bridge::BridgeError;

pub struct CoapBridge {
    endpoints: Vec<String>,
}

impl CoapBridge {
    pub fn new(endpoints: Vec<String>) -> Self {
        Self { endpoints }
    }

    pub async fn transmit(&self, orb: &OrbPayload) -> Result<(), BridgeError> {
        let data = orb.to_bytes();

        for url in &self.endpoints {
            let mut client = CoAPClient::from_url(url)
                .map_err(|e| BridgeError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

            client.post_with_timeout(&data, std::time::Duration::from_secs(5))
                .map_err(|e| BridgeError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

            println!("[CoAP] Orb {:?} transmitted to {}", orb.orb_id, url);
        }

        Ok(())
    }
}
