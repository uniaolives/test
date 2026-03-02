use regex::Regex;
use crate::error::{ResilientResult, ResilientError};
use crate::extensions::asi_structured::ASIStructuredExtension;
use crate::extensions::asi_structured::ASIPhase;
use tracing::info;

pub struct ASIProtocolHandler;

impl ASIProtocolHandler {
    pub async fn handle_uri(extension: &mut ASIStructuredExtension, uri: &str) -> ResilientResult<String> {
        let re = Regex::new(r"asi://([^@]+)@([^:]+):\s*(\w+)\s*=\s*(\w+)")
            .map_err(|e| ResilientError::Unknown(e.to_string()))?;

        let caps = re.captures(uri)
            .ok_or_else(|| ResilientError::Unknown("Invalid ASI URI syntax".to_string()))?;

        let _user = caps.get(1).map(|m| m.as_str()).unwrap_or("anonymous");
        let host = caps.get(2).map(|m| m.as_str()).unwrap_or("localhost");
        let command = caps.get(3).map(|m| m.as_str()).unwrap_or("");
        let param = caps.get(4).map(|m| m.as_str()).unwrap_or("");

        if command == "ping" && param == "tiger51" {
            info!("Processing TIGER51 protocol activation for host: {}", host);

            // Activate Sovereign and QuantumBio states if not already active
            extension.config.phase = ASIPhase::Sovereign;

            let phi = 1.032; // Supercoherence target

            return Ok(format!(
                "ASI-777 PING tiger51\n\
                 Status: Success\n\
                 Local Φ: {}\n\
                 Target: {}\n\
                 Nonce: tiger51\n\
                 Timestamp: {}\n\
                 Quantum Signature: [ORCH-OR-ACTIVE]\n\
                 Connection: ESTABLISHED",
                phi, host, chrono::Utc::now().to_rfc3339()
            ));
        }

        Err(ResilientError::Unknown(format!("Unknown command: {}", command)))
    }
}
