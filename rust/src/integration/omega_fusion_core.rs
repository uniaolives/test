// rust/src/integration/omega_fusion_core.rs

use crate::governance::archetype_core::{ArchetypeEnforcementUnit, ArchetypeViolation};
use crate::governance::matrix::{CompleteGovernanceMatrix, ActionContext, GovernanceDecision, ProposedAction};
use crate::mesh_neuron::nfg::BciFederatedNode;
use crate::security::physics_hypervisor::PhysicsHypervisor;
use crate::security::neural_firewall::NeuralFirewall;
use crate::security::reality_anchoring::{RealityWatermark, Scene, RealityAnchor};
use crate::security::view_uniqueness::CryptographicallyUniqueView;
use crate::bio_layer::biological_rate_limiter::BiologicalRateLimiter;
use crate::bio_layer::dopamine_monitor::DopamineSatietyMonitor;
use crate::bio_layer::private_inference::PrivateNeuralInference;
use crate::governance::archetype_core::AGICommand;

pub struct OmegaFusionCore {
    pub archetype_enforcer: ArchetypeEnforcementUnit,
    pub neural_federated_guard: BciFederatedNode,
    pub physics_hypervisor: PhysicsHypervisor,
    pub neural_firewall: NeuralFirewall,
    pub biological_rate_limiter: BiologicalRateLimiter,
    pub reality_watermarker: RealityWatermark,
    pub unique_view_generator: CryptographicallyUniqueView,
    pub safeword_anchor: RealityAnchor,
    pub dopamine_monitor: DopamineSatietyMonitor,
    pub private_inference: PrivateNeuralInference,
    pub governance: CompleteGovernanceMatrix,
}

pub struct AGIIntent {
    pub content: String,
}

impl AGIIntent {
    pub fn to_command(&self) -> AGICommand {
        AGICommand { content: self.content.clone() }
    }
}

pub struct UserContext {
    pub id: String,
    pub cognitive_coherence: f64,
}

pub struct ValidatedExperience {
    pub content: String,
}

#[derive(Debug)]
pub enum FusionError {
    InsufficientPhi(f64),
    CoreArchetypeViolation,
    GovernanceRejection(String),
    FirewallError(String),
    PhysicsViolation,
    SafewordActivated,
}

const PHI_MINIMUM_PRODUCTION: f64 = 0.72;

impl OmegaFusionCore {
    pub fn process_full_interaction(
        &mut self,
        agi_intent: AGIIntent,
        user: &UserContext,
    ) -> Result<ValidatedExperience, FusionError> {
        // 0. Pre-filtering
        if user.cognitive_coherence < PHI_MINIMUM_PRODUCTION {
            return Err(FusionError::InsufficientPhi(user.cognitive_coherence));
        }

        // 1. Archetype Verification
        self.archetype_enforcer.validate_command(agi_intent.to_command())
            .map_err(|e| match e {
                ArchetypeViolation::CoreViolation(_) => FusionError::CoreArchetypeViolation,
                _ => FusionError::CoreArchetypeViolation,
            })?;

        // 2. Neural Firewall
        self.neural_firewall.process_agi_command(crate::security::neural_firewall::NeuralCommand {})
            .map_err(|e| FusionError::FirewallError(e))?;

        // 3. Reality Integrity
        let _watermarked = self.reality_watermarker.embed_watermark(Scene {});

        // 4. Safeword Check
        if self.safeword_anchor.safeword_triggered() {
            self.safeword_anchor.activate_hardware_cutoff();
            return Err(FusionError::SafewordActivated);
        }

        // 5. Governance
        let context = ActionContext {
            user: user.id.clone(),
            phi: user.cognitive_coherence,
        };
        let action = ProposedAction { id: "1".into(), content: agi_intent.content.clone() };

        let decision = self.governance.evaluate_action(&action, &context);

        match decision {
            GovernanceDecision::Reject(reason) => return Err(FusionError::GovernanceRejection(reason)),
            _ => {}
        }

        Ok(ValidatedExperience { content: "OK".to_string() })
    }
}
