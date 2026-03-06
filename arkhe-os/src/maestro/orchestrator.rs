use crate::maestro::spine::{MaestroSpine, PsiState};
use crate::maestro::causality::BranchingEngine;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct MaestroOrchestrator {
    pub spine: MaestroSpine,
    pub engine: Arc<RwLock<BranchingEngine>>,
}

impl MaestroOrchestrator {
    pub fn new(spine: MaestroSpine, engine: Arc<RwLock<BranchingEngine>>) -> Self {
        Self { spine, engine }
    }

    pub async fn process_intent(&self, intent: &str, psi_state: &PsiState) -> Result<String, String> {
        // 1. Execute handover via Spine
        let response = self.spine.execute_handover(intent, psi_state).await?;

        // 2. Fork a new branch for this response
        let mut engine = self.engine.write().await;
        let _branch_id = engine.fork(None, response.clone(), psi_state.current_coherence);

        // 3. Prune for Novikov consistency
        engine.prune_branches();

        Ok(response)
    }
}
