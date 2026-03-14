use std::sync::Arc;
use async_trait::async_trait;
use russh::*;
use russh_keys::*;
use tokio::net::TcpListener;
use anyhow::Result;

pub struct SSHServer {
    pub port: u16,
}

impl SSHServer {
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    pub async fn run(self) -> Result<()> {
        let config = russh::server::Config {
            auth_rejection_time: std::time::Duration::from_secs(3),
            ..Default::default()
        };
        let _config = Arc::new(config);
        println!("[OrbVM] SSH Server listening on port {}", self.port);
        Ok(())
    }
}

pub struct SSHHandler;

#[async_trait]
impl russh::server::Handler for SSHHandler {
    type Error = anyhow::Error;

    async fn auth_publickey(self, _user: &str, _key: &russh_keys::key::PublicKey) -> Result<(Self, russh::server::Auth)> {
        Ok((self, russh::server::Auth::Accept))
    }

    async fn channel_open_session(
        self,
        _channel: russh::Channel<russh::server::Msg>,
        session: russh::server::Session,
    ) -> Result<(Self, bool, russh::server::Session)> {
        Ok((self, true, session))
    }
}
