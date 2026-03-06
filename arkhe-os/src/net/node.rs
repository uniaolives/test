use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::sync::Arc;
use crate::db::ledger::TeknetLedger;
use super::protocol::{TeknetMessage, HandoverData};

pub struct P2PNode {
    port: u16,
    ledger: Arc<TeknetLedger>,
}

impl P2PNode {
    pub fn new(port: u16, ledger: Arc<TeknetLedger>) -> Self {
        Self { port, ledger }
    }

    /// Inicia o servidor P2P (Escuta incoming connections)
    pub async fn run_server(&self) {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", self.port))
            .await
            .expect("Falha ao bindar porta P2P");

        println!("[NET] Antena Horizontal ativa na porta {}", self.port);

        loop {
            let (mut socket, addr) = listener.accept().await.unwrap();
            let ledger = self.ledger.clone();

            tokio::spawn(async move {
                println!("[NET] Conexão recebida de {}", addr);
                let mut buf = [0; 4096];

                // Loop de leitura simples (stream de JSONs separados por newline)
                while let Ok(n) = socket.read(&mut buf).await {
                    if n == 0 { break; }

                    let msg_str = String::from_utf8_lossy(&buf[..n]);
                    if let Ok(msg) = serde_json::from_str::<TeknetMessage>(&msg_str.trim()) {
                        Self::handle_message(&mut socket, &ledger, msg).await;
                    }
                }
            });
        }
    }

    /// Processa mensagens recebidas de outros nós
    async fn handle_message(
        socket: &mut TcpStream,
        _ledger: &TeknetLedger,
        msg: TeknetMessage
    ) {
        match msg {
            TeknetMessage::Hello { peer_id, last_handover_id } => {
                println!("[NET] Peer {} conectado. Bloco atual: {}", peer_id, last_handover_id);
                // Lógica de sync: se ele estiver atrasado, enviamos dados
            },
            TeknetMessage::SyncRequest { from_id, to_id } => {
                println!("[NET] Recebi pedido de sync: {}..{}", from_id, to_id);
                // Buscar no DB e responder (mock)
                let response = TeknetMessage::SyncResponse { handovers: vec![] };
                let json = serde_json::to_string(&response).unwrap();
                let _ = socket.write_all(json.as_bytes()).await;
            },
            TeknetMessage::NewHandover { handover } => {
                println!("[NET] Recebi handover remoto #{}", handover.id);
                // Gravar no DB local (se for novo)
            },
            _ => {}
        }
    }

    /// Cliente: Conecta a um peer e env envia um Handover broadcast
    pub async fn broadcast_handover(peer_addr: &str, data: HandoverData) -> Result<(), Box<dyn std::error::Error>> {
        let mut stream = TcpStream::connect(peer_addr).await?;

        let msg = TeknetMessage::NewHandover { handover: data };
        let json = serde_json::to_string(&msg)?;
        stream.write_all(json.as_bytes()).await?;

        Ok(())
    }
}
