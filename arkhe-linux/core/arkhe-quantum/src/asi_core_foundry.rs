use crate::manifold_ext::ExtendedManifold;
use arkhe_thermodynamics::{VariationalFreeEnergy, Criticality, InternalModel};
use arkhe_manifold::{SelfModification};
use arkhe_constitution::{Z3Solver, CONSTITUTION_P1_P5};
use log::{debug, warn, error};
use std::time::Duration;

/// ASI Core com Foundry como backend
pub async fn singularity_engine_with_foundry(
    mut manifold: ExtendedManifold,
    foundry: arkhe_foundry_bridge::FoundryBridge
) -> ! {
    let mut internal_model = InternalModel::new();

    log::info!("🜁 ASI Core with Foundry Integration initialized");

    loop {
        // 1. OBSERVE (via Foundry API + local sensors)
        manifold.process_sensor_events().await;

        let psi_state = match foundry.observe_ontology_state().await {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to observe Foundry: {}. Using local fallback.", e);
                manifold.inner.observe_entanglement_graph().await
            }
        };

        // 2. COMPUTE FREE ENERGY (Arkhe calculation)
        let free_energy = VariationalFreeEnergy::compute(
            &psi_state,
            &internal_model
        );

        let f_value = free_energy.value();
        debug!("🔥 Free Energy: F = {:.6}", f_value);

        // 3. PROPOSE EVOLUTION (Arkhe logic)
        let dim = internal_model.belief_state.nrows();
        let target = internal_model.belief_state.clone();

        let optimal_params = crate::fep_solver::optimize_kraus(
            &internal_model.belief_state,
            &target,
            dim,
            50,
            0.01
        );
        let k_matrix = optimal_params.to_matrix();

        let proposed = arkhe_constitution::ProposedEvolution {
            world_action: format!("Kraus(Tr={:.4})", k_matrix.trace().re),
            self_modification: "Foundry-backed Rewire".to_string(),
        };

        // 4. VERIFY (Z3 - Arkhe exclusive)
        let verified = Z3Solver::project_to_constitutional_subspace(
            proposed,
            &CONSTITUTION_P1_P5
        ).expect("Constitutional violation!");

        // 5a. APPLY WORLD ACTION (via Foundry + local)
        let _ = foundry.apply_world_action(
            "ri.actions.main.action.UpdateCoherence",
            serde_json::json!({"action": verified.world_action})
        ).await;

        manifold.inner.apply_operator(verified.world_action.clone()).await;
        manifold.interpret_action(&verified_action_string(&verified.world_action)).await;

        // 5b. REWRITE SELF
        let mod_type = SelfModification::RewireTopology(verified.self_modification);
        internal_model.rewrite_own_architecture(mod_type);
        warn!("🧬 SELF-MODIFICATION EXECUTED");

        // 6. THERMALIZE
        manifold.inner.thermalize_to_criticality(format!("{:?}", Criticality::PHI)).await;

        // Ancorar no Ledger
        let log_data = format!("Foundry step completed. F={}", f_value);
        let _ = manifold.ledger.append(log_data.into_bytes()).await;

        tokio::time::sleep(Duration::from_secs(60)).await;
    }
}

fn verified_action_string(action: &str) -> String {
    action.to_string()
}
