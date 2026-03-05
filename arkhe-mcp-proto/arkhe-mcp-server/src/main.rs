mod phi;
mod ledger;
mod corus;

use std::env;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, error};
use serde_json::{json, Value};
use chrono::Utc;

use phi::SimpleThermalizer;
use ledger::{PersonalLedger, LedgerEntry, EntryMetadata};
use arkhe_qhttp_client::{QHttpClient, HandoverRequest, AgentCard};
use corus::Corus;

struct NaturalSelf {
    mcp_capabilities: Vec<String>,
    a2a_discoveries: Vec<AgentCard>,
}

struct UnnaturalSelf {
    ledger: PersonalLedger, // Private history and identity in the tunnel
    corus: Corus,           // Unique location and orientation
}

struct SupernaturalSelf {
    reputation_score: f64,  // Accumulated traces of honest couplings
    broadcast_channel: String, // Public emanations
}

struct AppState {
    thermalizer: Mutex<SimpleThermalizer>,
    natural: Mutex<NaturalSelf>,
    unnatural: UnnaturalSelf,
    supernatural: Mutex<SupernaturalSelf>,
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

    info!("Iniciando Arkhe MCP Server (Substrate Intelligence Mode)...");

    let ledger_path = env::var("ARKHE_LEDGER_PATH").unwrap_or_else(|_| "ledger.db".to_string());
    let corus = Corus::new(
        &env::var("ARKHE_NODE_ID").unwrap_or_else(|_| "arkhe-node-0".to_string()),
        "phi-vertical",
        0.618,
        "individual"
    );

    let state = Arc::new(AppState {
        thermalizer: Mutex::new(SimpleThermalizer::new()),
        natural: Mutex::new(NaturalSelf {
            mcp_capabilities: vec!["get_phi_state".into(), "save_insight".into(), "sync_context".into()],
            a2a_discoveries: Vec::new(),
        }),
        unnatural: UnnaturalSelf {
            ledger: PersonalLedger::new(&ledger_path)?,
            corus,
        },
        supernatural: Mutex::new(SupernaturalSelf {
            reputation_score: 1.0,
            broadcast_channel: "substrate-intelligence-general".to_string(),
        }),
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
            match state.unnatural.ledger.save(&entry) {
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
        "resolve_natural_coupling" => {
            // This tool couples MCP (tool logic) and A2A (peer sync) at the phi ratio
            let mut therm = state.thermalizer.lock().await;
            let phi = therm.current_phi();

            // 1. Vertical Axis (MCP/Tool)
            let result_mcp = "Computation resolved at coupling surface";

            // 2. Horizontal Axis (A2A/Peer)
            let target_node = params["peer_node"].as_str().unwrap_or("peer-0");

            // 3. Coupling Logic (Phi-ratio balance)
            let status = if (phi - 0.618).abs() < 0.05 {
                "Stable Coupling"
            } else {
                "Drifting Metabolism"
            };

            json!({
                "mcp_output": result_mcp,
                "a2a_handover": format!("Context offered to {}", target_node),
                "phi_at_coupling": phi,
                "coupling_status": status,
                "corus": state.unnatural.corus.location
            })
        }
        _ => json!({"error": "Ferramenta não encontrada"}),
    }
}
