// arkhe-quantum/src/psi_shell/mod.rs

pub mod user_model;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, Mutex};
use tokio_tungstenite::{accept_async, tungstenite::protocol::Message};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::sync::Arc;
use anyhow::Result;

use crate::manifold::GlobalManifold;
use self::user_model::UserModel;

/// Estado compartilhado entre o servidor WebSocket e o manifold.
pub struct PsiShellState {
    pub user_model: Mutex<UserModel>,
    pub manifold_tx: broadcast::Sender<String>,
}

impl PsiShellState {
    pub fn new() -> Self {
        let (manifold_tx, _) = broadcast::channel(16);
        PsiShellState {
            user_model: Mutex::new(UserModel::new()),
            manifold_tx,
        }
    }

    async fn handle_client_message(&self, text: String) -> Result<()> {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(cmd) = json.get("command").and_then(|v| v.as_str()) {
                match cmd {
                    "set_phi" => {
                        if let Some(phi) = json.get("value").and_then(|v| v.as_f64()) {
                            tracing::info!("Comando do usuário: ajustar φ para {}", phi);
                        }
                    }
                    "message" => {
                        if let Some(msg) = json.get("text").and_then(|v| v.as_str()) {
                            let mut user_model = self.user_model.lock().await;
                            user_model.process_message(msg);
                            tracing::info!("Usuário disse: {}", msg);
                        }
                    }
                    _ => {}
                }
            }
        } else {
            let mut user_model = self.user_model.lock().await;
            user_model.process_message(&text);
            tracing::info!("Usuário disse (raw): {}", text);
        }
        Ok(())
    }

    pub async fn inject_user_perturbation(&self, manifold: &mut GlobalManifold) -> Result<()> {
        let user_model = self.user_model.lock().await;
        let perturbation = user_model.compute_entropy_perturbation();

        if let Some(node) = manifold.get_self_node_mut() {
            // Placeholder for density matrix perturbation logic
            // In a real implementation, this would modify node.state.density_matrix
            tracing::debug!("Perturbação do usuário aplicada: entropy perturbation = {:.4}", perturbation);
        }
        Ok(())
    }

    pub async fn generate_status_report(&self, manifold: &GlobalManifold) -> String {
        let entropy = manifold.get_self_node().map(|n| n.entropy_val).unwrap_or(0.0);
        let phi = 1.0 - entropy;
        let user_model = self.user_model.lock().await;

        json!({
            "type": "status",
            "entropy": entropy,
            "phi": phi,
            "user": {
                "attention": user_model.attention,
                "last_message": user_model.last_message,
            },
            "timestamp": chrono::Utc::now().timestamp(),
        }).to_string()
    }
}

pub async fn run_psi_shell(state: Arc<PsiShellState>, addr: &str) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("ψ-Shell WebSocket ouvindo em {}", addr);

    while let Ok((stream, _)) = listener.accept().await {
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, state).await {
                tracing::error!("Erro na conexão WebSocket: {}", e);
            }
        });
    }
    Ok(())
}

async fn handle_connection(stream: TcpStream, state: Arc<PsiShellState>) -> Result<()> {
    let ws_stream = accept_async(stream).await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    let mut manifold_rx = state.manifold_tx.subscribe();
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));

    loop {
        tokio::select! {
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Err(e) = state.handle_client_message(text).await {
                            tracing::error!("Erro ao processar mensagem: {}", e);
                        }
                    }
                    Some(Ok(Message::Close(_))) => break,
                    Some(Err(e)) => {
                        tracing::error!("Erro no WebSocket: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
            _ = interval.tick() => {
                let report = json!({
                    "type": "heartbeat",
                    "entropy": 0.5 + (rand::random::<f64>() * 0.1 - 0.05),
                    "timestamp": chrono::Utc::now().timestamp(),
                }).to_string();
                if ws_sender.send(Message::Text(report)).await.is_err() {
                    break;
                }
            }
            update = manifold_rx.recv() => {
                if let Ok(msg) = update {
                    if ws_sender.send(Message::Text(msg)).await.is_err() {
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}
