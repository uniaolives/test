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
            let request_data = data.clone();
            // In coap 0.24, post is likely async if using tokio features,
            // but the error suggests it returns a Result directly.
            // Let's check the implementation again or try without await.
            // Wait, the error message said "is not a future", so it's synchronous.
            let _ = CoAPClient::post(url, request_data)
                .map_err(|e: std::io::Error| BridgeError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

            println!("[CoAP] Orb {:?} transmitted to {}", orb.orb_id, url);
        }

        Ok(())
    }
}
