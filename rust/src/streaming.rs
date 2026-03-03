// rust/src/streaming.rs
use crate::cge_log;

pub struct TwitchPlatform;

impl TwitchPlatform {
    pub fn new() -> Self {
        Self
    }
}

pub struct IPTVStreamProtocol;

impl IPTVStreamProtocol {
    pub async fn push_global(&self, message: String) {
        cge_log!(broadcast, "📡 IPTV: Push global: {}", message);
    }
}
