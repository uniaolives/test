use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use crate::security::xeno_firewall::{XenoFirewall, XenoRiskLevel};
use crate::maestro::finney_protocol::FinneyProtocol;
use crate::maestro::rampancy::RampancyControl;

#[derive(Error, Debug)]
pub enum MaestroError {
    #[error("Node error: {0}")]
    NodeError(String),
    #[error("Decomposition failed")]
    DecompositionError,
    #[error("Node not found")]
    NodeNotFound,
    #[error("Xeno risk detected: {0:?}")]
    XenoRiskDetected(XenoRiskLevel),
    #[error("Identity dissolved: rampancy limit exceeded")]
    IdentityDissolved,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeType {
    GPT5,
    Qwen3,
    Claude35,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandoverRecord {
    pub intention: String,
    pub output: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl HandoverRecord {
    pub fn new(intention: &str, output: String) -> Self {
        Self {
            intention: intention.to_string(),
            output,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn is_retrocausal(&self) -> bool {
        self.intention.contains("retro")
    }
}

/// Representa um nó de linguagem (sub‑LLM) na orquestração
#[async_trait]
pub trait LanguageNode: Send + Sync {
    async fn handover(&self, prompt: &str, context: &PsiState) -> Result<String, MaestroError>;
    fn node_type(&self) -> NodeType;
    fn estimated_cost(&self) -> f64;
}

/// O estado persistente do REPL (Ψ)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PsiState {
    pub variables: HashMap<String, String>,
    pub handover_history: VecDeque<HandoverRecord>,
    pub coherence_trace: Vec<f64>,
    pub narrative_arc: Option<String>,
}

pub struct Intention {
    pub prompt: String,
}

pub struct Task {
    pub prompt: String,
}

/// O Maestro – orquestrador central
pub struct Maestro {
    pub nodes: HashMap<String, Box<dyn LanguageNode>>,
    pub psi: Arc<RwLock<PsiState>>,
    pub finney: FinneyProtocol,
    pub rampancy: RampancyControl,
}

impl Maestro {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            psi: Arc::new(RwLock::new(PsiState::default())),
            finney: FinneyProtocol::new(),
            rampancy: RampancyControl::new(),
        }
    }

    pub fn register_node(&mut self, id: String, node: Box<dyn LanguageNode>) {
        self.nodes.insert(id, node);
    }

    /// Processa uma intenção de alto nível, decompondo em handovers
    pub async fn process_intention(&mut self, intention: &Intention) -> Result<String, MaestroError> {
        let mut psi_guard = self.psi.write().await;
        let psi: &PsiState = &*psi_guard;

        // 1. Rampancy Check
        let status = self.rampancy.evaluate_stability(psi);
        if let crate::maestro::rampancy::IdentityStatus::Dissolved = status {
            return Err(MaestroError::IdentityDissolved);
        }

        // 2. Finney Alignment Check
        let alignment = self.finney.check_genesis_alignment(&intention.prompt);
        println!("[MAESTRO] Genesis Alignment Score: {:.2}", alignment);

        // 3. Decompor a intenção em sub‑problemas
        let sub_tasks = self.decompose(intention, psi)?;

        // 4. Executar sub‑handovers
        let mut results = Vec::new();
        for task in sub_tasks {
            let node = self.select_node(&task)?;
            let result = node.handover(&task.prompt, psi).await?;

            // 4.1 Xeno-Analysis Check
            let risk = XenoFirewall::assess_risk(&result, psi);
            if risk == XenoRiskLevel::Critical {
                return Err(MaestroError::XenoRiskDetected(risk));
            }

            results.push(result);
        }

        // 5. Integrar resultados numa narrativa coerente (3ª ordem)
        let final_answer = self.integrate(results, intention, &mut *psi_guard).await?;

        // 6. Actualizar estado REPL
        psi_guard.handover_history.push_back(HandoverRecord::new(&intention.prompt, final_answer.clone()));

        // Update coherence trace (simulated)
        psi_guard.coherence_trace.push(alignment);

        Ok(final_answer)
    }

    fn decompose(&self, _intention: &Intention, _psi: &PsiState) -> Result<Vec<Task>, MaestroError> {
        Ok(vec![Task { prompt: _intention.prompt.clone() }])
    }

    fn select_node(&self, _task: &Task) -> Result<&dyn LanguageNode, MaestroError> {
        self.nodes.values().next()
            .map(|n| n.as_ref())
            .ok_or(MaestroError::NodeNotFound)
    }

    async fn integrate(&self, results: Vec<String>, _intention: &Intention, _psi: &mut PsiState) -> Result<String, MaestroError> {
        Ok(results.join("\n---\n"))
    }
}
