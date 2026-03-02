// rust/src/runtime/nullclaw.rs
use crate::error::{ResilientError, ResilientResult};
use crate::runtime::backend::{RuntimeBackend, BackendConfig};
use crate::runtime::context_manager::ContextWindow;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct HandoverPacket {
    input: String,
    context: String,
}

pub struct NullClawBackend {
    config: Option<BackendConfig>,
    client: reqwest::blocking::Client,
}

impl NullClawBackend {
    pub fn new() -> Self {
        Self {
            config: None,
            client: reqwest::blocking::Client::new(),
        }
    }
}

impl RuntimeBackend for NullClawBackend {
    fn initialize(&mut self, config: BackendConfig) -> ResilientResult<()> {
        self.config = Some(config);
        Ok(())
    }

    fn process(&self, input: &str, _context: &ContextWindow) -> ResilientResult<String> {
        let config = self.config.as_ref().ok_or(ResilientError::Configuration(
            "NullClawBackend not initialized".to_string(),
        ))?;

        let packet = HandoverPacket {
            input: input.to_string(),
            context: "".to_string(), // Simplified context handover
        };

        let response = self.client.post(&config.endpoint)
            .header("Authorization", format!("Bearer {}", config.api_key.as_deref().unwrap_or("")))
            .json(&packet)
            .send()
            .map_err(|e| ResilientError::RuntimeBackend(format!("NullClaw request failed: {}", e)))?;

        if response.status().is_success() {
            let body = response.text().map_err(|e| ResilientError::RuntimeBackend(format!("Failed to read response: {}", e)))?;
            Ok(body)
        } else {
            Err(ResilientError::RuntimeBackend(format!("NullClaw returned error: {}", response.status())))
        }
    }

    fn get_name(&self) -> String {
        "NullClaw Runtime (Zig)".to_string()
    }
}
