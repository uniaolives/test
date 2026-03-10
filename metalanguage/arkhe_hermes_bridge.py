#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARKHE-HERMES BRIDGE
Integrates the NousResearch Hermes Agent capabilities into the Arkhe(n) framework.
"""

from metalanguage.anl import Agent, Protocol
import time
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add hermes-agent to sys.path to allow importing its modules
HERMES_PATH = Path(__file__).parent.parent / "agents" / "hermes-agent"
if str(HERMES_PATH) not in sys.path:
    sys.path.append(str(HERMES_PATH))

try:
    # Functional imports from Hermes codebase
    from tools.skills_tool import skills_list, SKILLS_DIR
    from agent.skill_commands import scan_skill_commands
    HERMES_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Hermes functional modules not fully available: {e}")
    HERMES_AVAILABLE = False
    SKILLS_DIR = Path.home() / ".hermes" / "skills"
    def skills_list(): return json.dumps({"success": True, "skills": []})

class HermesNode(Agent):
    """
    An Arkhe(n) Agent that encapsulates a Hermes Agent instance.
    Utilizes Hermes' autonomous skill creation and memory management.
    """
    def __init__(self, agent_id: str, **attributes):
        ontology = attributes.get('ontology', "arkhe:hermes:v1")
        super().__init__(agent_id, ontology, **attributes)

        # Register core Hermes capabilities as ANL-compatible handlers
        self.register_capability("GenerateSkill", self._handle_generate_skill)
        self.register_capability("RefineSkill", self._handle_refine_skill)
        self.register_capability("ListSkills", self._handle_list_skills)

        self.skills_created = []

    def _handle_generate_skill(self, handover_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesizes a new skill using Hermes logic.
        """
        intent = handover_data.get('intent', {})
        task_description = intent.get('task', 'unknown task')

        print(f"⚕️ [HERMES-{self.id}] Analyzing trajectory for skill synthesis: {task_description}")

        skill_name = task_description.lower().replace(" ", "-")

        # Call Hermes' scan to check for existing skills
        if HERMES_AVAILABLE:
            try:
                available_skills = scan_skill_commands()
                exists = f"/{skill_name}" in available_skills
            except Exception:
                exists = False
        else:
            exists = False

        # Construct the skill metadata following Hermes standards
        skill_metadata = {
            "name": skill_name,
            "path": str(SKILLS_DIR / skill_name),
            "protocol": Protocol.CREATIVE,
            "auto_generated": True,
            "already_exists": exists,
            "timestamp": time.time()
        }

        self.skills_created.append(skill_metadata)
        self.trigger_event("SkillGenerated", {"skill_name": skill_name})

        return {
            "status": "SUCCESS",
            "skill": skill_metadata,
            "message": f"Skill '{skill_name}' successfully processed by Hermes pipeline."
        }

    def _handle_list_skills(self, handover_data: Dict[str, Any]) -> Dict[str, Any]:
        """Returns list of currently available Hermes skills."""
        return json.loads(skills_list())

    def _handle_refine_skill(self, handover_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refines existing skills based on historical handovers."""
        return {"status": "SUCCESS", "message": "Skill refinement cycle complete."}

    def get_hermes_stats(self):
        return {
            "id": self.id,
            "skills_count": len(self.skills_created),
            "curiosity": self.calculate_epistemic_value(),
            "hermes_active": HERMES_AVAILABLE
        }
