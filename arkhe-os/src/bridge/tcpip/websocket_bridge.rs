// arkhe-os/src/bridge/tcpip/websocket_bridge.rs

use tokio_tungstenite::{connect_async, WebSocketStream, MaybeTlsStream};
use tokio::net::TcpStream;
use futures::{SinkExt, StreamExt};
use crate::orb::core::OrbPayload;
use tokio_tungstenite::tungstenite::Message;

pub struct WebSocketBridge {
    connections: Vec<WebSocketStream<MaybeTlsStream<TcpStream>>>,
}

impl WebSocketBridge {
    /// Conecta a múltiplos endpoints WebSocket
    pub async fn connect(urls: &[&str]) -> Self {
        let mut connections = Vec::new();

        for url in urls {
            if let Ok((ws_stream, _)) = connect_async(*url).await {
                connections.push(ws_stream);
            }
        }

        Self { connections }
    }

    /// Transmite Orb em tempo real
    pub async fn broadcast(&mut self, orb: &OrbPayload) {
        let payload = orb.to_bytes();
        let message = Message::Binary(payload.into());

        for conn in &mut self.connections {
            let _ = conn.send(message.clone()).await;
        }
    }
}
