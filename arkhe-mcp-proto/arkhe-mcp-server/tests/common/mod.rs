use serde_json::{json, Value};

pub struct TestMcpServer {
}

impl TestMcpServer {
    pub async fn start_test_mcp_server() -> Self {
        Self {}
    }

    pub async fn call_tool(&self, name: &str, params: Value) -> Value {
        match name {
            "get_phi_state" => json!({
                "phi": 0.618,
                "regime": "critical (golden)"
            }),
            "set_phi_state" => json!({
                "new_phi": params["target_phi"].as_f64().unwrap_or(0.618),
                "status": "updated"
            }),
            "save_insight" => json!("Insight salvo com sucesso"),
            "search_memory_with_phi" => json!({
                "results": [
                    {
                        "content": "Insight de teste integração",
                        "score": 0.95
                    }
                ]
            }),
            "sync_context_to_quantum" => json!({
                "status": "synced",
                "handover_id": "test-123",
                "phi_remote": 0.615,
            }),
            _ => json!({"error": "unknown tool"})
        }
    }
}
