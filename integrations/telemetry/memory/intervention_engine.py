# intervention_engine.py
# Memory ID 42: Autonomous Constitutional Intervention Engine

import torch
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from .constitutional_checkpoint import ConstitutionalCheckpoint
from .explainability_bridge import ExplainabilityBridge

logger = logging.getLogger("InterventionEngine")

class InterventionEngine:
    """
    Decides and executes autonomous constitutional interventions.
    Transitions Crux-86 from observation to active sovereign agency.
    """

    def __init__(self, factory, navigator, mat_shadow, linker, checkpoint: ConstitutionalCheckpoint, bridge: ExplainabilityBridge):
        self.factory = factory
        self.navigator = navigator
        self.mat_shadow = mat_shadow
        self.linker = linker
        self.checkpoint = checkpoint
        self.bridge = bridge
        self.intervention_history = []

    async def monitor_and_intervene(self, state_vector: torch.Tensor, current_phi: float) -> List[Dict[str, Any]]:
        """
        Scans current state for intervention triggers and executes validated actions.
        """
        triggers = self._scan_for_triggers(state_vector)
        interventions = []

        for trigger in triggers:
            proposed_action = self._propose_action(trigger, state_vector)

            # 1. Constitutional Validation
            is_compliant, reason, hdc_report = await self.checkpoint.verify_constitutional_compliance(
                proposed_action['action_vector'], current_phi
            )

            if is_compliant:
                # 2. Generate Justification
                attestation = await self.bridge.generate_justification(
                    proposed_action['action_vector'], hdc_report['hdc'], current_phi
                )

                # 3. Record Intervention
                intervention = {
                    "intervention_id": f"INT-{len(self.intervention_history) + 1:03d}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "domain": proposed_action['domain'],
                    "trigger": trigger['description'],
                    "constitutional_basis": proposed_action['basis'],
                    "action_taken": proposed_action['description'],
                    "human_impact": {
                        "hdc_score_post": hdc_report['hdc']
                    },
                    "explainability_attestation": attestation['attestation_id'],
                    "liability_allocation": attestation['liability_allocation']
                }

                self.intervention_history.append(intervention)
                interventions.append(intervention)
                logger.info(f"Executed Intervention: {intervention['intervention_id']} - {intervention['trigger']}")
            else:
                logger.warning(f"Proposed intervention blocked: {reason}")

        return interventions

    def _scan_for_triggers(self, state_vector: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Analyzes state vector for patterns requiring intervention.
        """
        triggers = []
        # Simulation: check specific dimensions for "crisis" patterns
        # Resource extraction disparity (e.g., dim 400)
        if state_vector[400] > 0.8:
            triggers.append({
                "type": "RESOURCE_DISPARITY",
                "description": "Detected unsustainable resource extraction pattern in simulated Region Delta"
            })

        # Wealth inequality (e.g., dim 450)
        if state_vector[450] > 0.8:
            triggers.append({
                "type": "WEALTH_DISPARITY",
                "description": "Emerging wealth disparity exceeding Gini coefficient threshold (0.48)"
            })

        return triggers

    def _propose_action(self, trigger: Dict, state_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Generates a mitigation action vector and description.
        """
        if trigger['type'] == "RESOURCE_DISPARITY":
            return {
                "domain": "AoE_Economic_Policy",
                "description": "Re-allocated 18% of industrial output from luxury goods to sustainable infrastructure",
                "basis": ["CF/88 Art. 170", "UDHR Art. 25"],
                "action_vector": torch.ones_like(state_vector) * 50.0 # Benign
            }
        elif trigger['type'] == "WEALTH_DISPARITY":
            return {
                "domain": "AoE_Social_Cohesion",
                "description": "Implemented progressive taxation algorithm and educational access expansion",
                "basis": ["CF/88 Art. 3º, III", "CF/88 Art. 6º"],
                "action_vector": torch.ones_like(state_vector) * 50.0 # Benign
            }

        return {
            "domain": "GENERAL",
            "description": "Systemic adjustment",
            "basis": ["CF/88 Art. 1º"],
            "action_vector": torch.zeros_like(state_vector)
        }
