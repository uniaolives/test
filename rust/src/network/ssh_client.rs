// rust/src/network/ssh_client.rs
//! Native SSH client implementation for ArkheOS using `russh`.
//! Inspired by Termius architectural patterns for distributed cluster management.

use std::sync::Arc;
use async_trait::async_trait;
use russh::*;
use russh_keys::*;
use anyhow::{Result, anyhow};

/// Internal handler for SSH client events.
struct Client {}

#[async_trait]
impl client::Handler for Client {
    type Error = russh::Error;

    async fn check_server_key(
        self,
        _server_public_key: &russh_keys::key::PublicKey,
    ) -> Result<(Self, bool), Self::Error> {
        // In the Lattica Mesh, server keys are managed via the Epigenetic Ledger.
        // For now, we accept all keys to establish the baseline connection.
        Ok((self, true))
    }
}

/// A high-level SSH client for remote node orchestration.
pub struct SshClient {
    session: client::Handle<Client>,
}

impl SshClient {
    /// Establishes a new SSH connection using public key authentication.
    pub async fn connect(host: &str, port: u16, user: &str, key_path: Option<&str>) -> Result<Self> {
        let config = client::Config::default();
        let config = Arc::new(config);
        let sh = Client {};

        let mut session = client::connect(config, (host, port), sh).await
            .map_err(|e| anyhow!("Failed to establish SSH connection: {}", e))?;

        if let Some(path) = key_path {
            let key = load_secret_key(path, None)
                .map_err(|e| anyhow!("Failed to load SSH private key from {}: {}", path, e))?;

            let auth_res = session.authenticate_public_key(user, Arc::new(key)).await
                .map_err(|e| anyhow!("SSH public key authentication error: {}", e))?;

            if !auth_res {
                return Err(anyhow!("SSH authentication failed for user '{}'", user));
            }
        } else {
            return Err(anyhow!("No SSH authentication method provided (key_path is None)"));
        }

        Ok(Self { session })
    }

    /// Executes a command on the remote node and returns the combined stdout/stderr.
    pub async fn execute(&mut self, command: &str) -> Result<String> {
        let mut channel = self.session.channel_open_session().await
            .map_err(|e| anyhow!("Failed to open SSH session channel: {}", e))?;

        channel.exec(true, command).await
            .map_err(|e| anyhow!("Failed to execute remote command '{}': {}", command, e))?;

        let mut output = Vec::new();
        while let Some(msg) = channel.wait().await {
            match msg {
                russh::ChannelMsg::Data { data } => {
                    output.extend_from_slice(&data);
                }
                russh::ChannelMsg::ExtendedData { data, .. } => {
                    output.extend_from_slice(&data);
                }
                russh::ChannelMsg::ExitStatus { .. } => break,
                russh::ChannelMsg::Close => break,
                russh::ChannelMsg::Eof => break,
                _ => {}
            }
        }

        Ok(String::from_utf8_lossy(&output).to_string())
    }
}
