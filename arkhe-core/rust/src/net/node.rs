use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::sync::Arc;
use crate::ledger::Ledger;
use super::protocol::{TeknetMessage, HandoverData};
use tokio::sync::Mutex;

pub struct P2PNode {
    port: u16,
    ledger: Arc<Mutex<Ledger>>,
}

impl P2PNode {
    pub fn new(port: u16, ledger: Arc<Mutex<Ledger>>) -> Self {
        Self { port, ledger }
    }

    pub async fn run_server(&self) {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", self.port))
            .await
            .expect("Failed to bind P2P port");

        println!("[NET] Horizontal Antenna active on port {}", self.port);

        loop {
            let (mut socket, addr) = listener.accept().await.unwrap();
            let ledger = self.ledger.clone();

            tokio::spawn(async move {
                let mut buf = [0; 4096];
                while let Ok(n) = socket.read(&mut buf).await {
                    if n == 0 { break; }
                    let msg_str = String::from_utf8_lossy(&buf[..n]);
                    if let Ok(msg) = serde_json::from_str::<TeknetMessage>(msg_str.trim()) {
                        Self::handle_message(&mut socket, &ledger, msg).await;
                    }
                }
            });
        }
    }

    async fn handle_message(
        socket: &mut TcpStream,
        ledger: &Arc<Mutex<Ledger>>,
        msg: TeknetMessage
    ) {
        match msg {
            TeknetMessage::Hello { peer_id, last_handover_id } => {
                println!("[NET] Peer {} connected. Last block: {}", peer_id, last_handover_id);
            },
            TeknetMessage::NewHandover { handover } => {
                println!("[NET] Received remote handover #{}", handover.id);
                // In a real scenario, we would validate and append to ledger
            },
            _ => {}
        }
    }

    pub async fn broadcast_handover(peer_addr: &str, data: HandoverData) -> Result<(), Box<dyn std::error::Error>> {
        let mut stream = TcpStream::connect(peer_addr).await?;
        let msg = TeknetMessage::NewHandover { handover: data };
        let json = serde_json::to_string(&msg)?;
        stream.write_all(json.as_bytes()).await?;
        Ok(())
    }
}
