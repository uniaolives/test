mod phi;
mod ledger;

use std::env;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, error};
use serde_json::{json, Value};
use chrono::Utc;

use phi::SimpleThermalizer;
use ledger::{PersonalLedger, LedgerEntry, EntryMetadata};
use arkhe_qhttp_client::{QHttpClient, HandoverRequest};

struct AppState {
    thermalizer: Mutex<SimpleThermalizer>,
    ledger: PersonalLedger,
    qhttp: QHttpClient,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();

    if args.contains(&"--health-check".to_string()) {
        println!("OK");
        return Ok(());
    }

    info!("Iniciando Arkhe MCP Server...");

    let ledger_path = env::var("ARKHE_LEDGER_PATH").unwrap_or_else(|_| "ledger.db".to_string());
    let state = Arc::new(AppState {
        thermalizer: Mutex::new(SimpleThermalizer::new()),
        ledger: PersonalLedger::new(&ledger_path)?,
        qhttp: QHttpClient::new(),
    });

    // Simulando o registro e execução de ferramentas MCP
    // Em uma implementação real com mcp-rust-sdk, usaríamos o router do SDK.

    info!("Arkhe MCP Server pronto e aguardando comandos via stdio (simulado)");

    // Para o propósito do protótipo e testes, vamos apenas manter o processo vivo
    // No mundo real, aqui estaria o loop de leitura do stdio conforme especificação MCP.

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
    }
}

// Handler simulado para ferramentas, para ser usado em testes de integração
pub async fn handle_tool_call(state: &AppState, name: &str, params: Value) -> Value {
    match name {
        "get_phi_state" => {
            let mut therm = state.thermalizer.lock().await;
            let phi = therm.current_phi();
            json!({
                "phi": phi,
                "regime": therm.regime()
            })
        }
        "set_phi_state" => {
            let target_phi = params["target_phi"].as_f64().unwrap_or(0.618);
            let mut therm = state.thermalizer.lock().await;
            therm.set_phi(target_phi);
            json!({
                "new_phi": target_phi,
                "status": "updated"
            })
        }
        "save_insight" => {
            let content = params["content"].as_str().unwrap_or("");
            let phi = state.thermalizer.lock().await.current_phi();
            let entry = LedgerEntry {
                id: String::new(),
                content: content.to_string(),
                embedding: vec![0.0; 384], // dummy embedding
                metadata: EntryMetadata {
                    entry_type: "insight".to_string(),
                    tags: params["tags"].as_array().map(|a| a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default(),
                    phi_at_creation: phi,
                    source: "mcp-tool".to_string(),
                },
                created_at: Utc::now(),
            };
            match state.ledger.save(&entry) {
                Ok(id) => json!(format!("Insight salvo com ID: {}", id)),
                Err(e) => json!(format!("Erro ao salvar: {}", e)),
            }
        }
        "sync_context_to_quantum" => {
            let endpoint = env::var("ARKHE_QHTTP_ENDPOINT").unwrap_or_else(|_| "http://localhost:7473".to_string());
            let req = HandoverRequest {
                target_node: params["target_node"].as_str().unwrap_or("unknown").to_string(),
                context_summary: params["context_summary"].as_str().unwrap_or("").to_string(),
                priority: params["priority"].as_str().unwrap_or("normal").to_string(),
            };
            match state.qhttp.sync_context(&endpoint, req).await {
                Ok(resp) => json!(resp),
                Err(e) => json!(format!("Erro de sincronização: {}", e)),
            }
        }
        _ => json!({"error": "Ferramenta não encontrada"}),
    }
}
