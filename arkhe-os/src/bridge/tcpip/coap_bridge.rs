// arkhe-os/src/bridge/tcpip/coap_bridge.rs

use coap::UdpCoAPClient;
use coap::CoAPClient;
use coap_lite::RequestType as Method;
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
            UdpCoAPClient::post(url, request_data).await
                .map_err(|e: std::io::Error| BridgeError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
            let mut client = CoAPClient::new(url)
                .map_err(|e| BridgeError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

            client.request_path("/", Method::Post, Some(data.clone()), None, None)
                .map_err(|e| BridgeError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

            println!("[CoAP] Orb {:?} transmitted to {}", orb.orb_id, url);
        }

        Ok(())
    }
}
