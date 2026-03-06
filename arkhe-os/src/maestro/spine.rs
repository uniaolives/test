use crate::maestro::api_wrapper::PTPApiWrapper;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsiState {
    pub repl_variables: HashMap<String, f64>,
    pub current_coherence: f64,
}

impl Default for PsiState {
    fn default() -> Self {
        let mut vars = HashMap::new();
        vars.insert("bio".to_string(), 0.5);
        vars.insert("aff".to_string(), 0.5);
        vars.insert("soc".to_string(), 0.5);
        vars.insert("cog".to_string(), 0.5);
        Self {
            repl_variables: vars,
            current_coherence: 1.0,
        }
    }
}

pub struct MaestroSpine {
    wrapper: PTPApiWrapper,
    endpoint: String,
}

impl MaestroSpine {
    pub fn new(wrapper: PTPApiWrapper, endpoint: &str) -> Self {
        Self {
            wrapper,
            endpoint: endpoint.to_string(),
        }
    }

    pub async fn execute_handover(&self, prompt: &str, psi_state: &PsiState) -> Result<String, String> {
        let payload = serde_json::json!({
            "model": "arkhe-maestro-v1",
            "messages": [
                {"role": "system", "content": format!("Ψ State: {:?}", psi_state.repl_variables)},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.28,
            "stream": false
        });

        let response = self.wrapper.execute_raw(&self.endpoint, payload).await?;

        let content = response["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("Failed to parse response content")?
            .to_string();

        Ok(content)
    }
}
