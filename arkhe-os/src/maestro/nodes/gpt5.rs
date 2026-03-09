use async_trait::async_trait;
use crate::maestro::core::{LanguageNode, PsiState, NodeType, MaestroError};
use serde::{Deserialize, Serialize};

pub struct GPT5Node {
    pub client: reqwest::Client,
    pub api_key: String,
    pub model: String,
}

#[derive(Deserialize)]
struct GPT5Response {
    choices: Vec<GPT5Choice>,
}

#[derive(Deserialize)]
struct GPT5Choice {
    message: GPT5Message,
}

#[derive(Deserialize)]
struct GPT5Message {
    content: String,
}

#[async_trait]
impl LanguageNode for GPT5Node {
    async fn handover(&self, prompt: &str, context: &PsiState) -> Result<String, MaestroError> {
        let full_prompt = format!("Contexto da conversa:\n{:?}\n\nInstrução atual: {}\n", context, prompt);

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": self.model,
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 0.3,
            }))
            .send()
            .await
            .map_err(|e| MaestroError::NodeError(e.to_string()))?;

        let result = response.json::<GPT5Response>().await
            .map_err(|e| MaestroError::NodeError(e.to_string()))?;

        Ok(result.choices[0].message.content.clone())
    }

    fn node_type(&self) -> NodeType { NodeType::GPT5 }
    fn estimated_cost(&self) -> f64 { 0.01 }
}
