import json
import logging
from typing import List, Dict, Any

class TLAMonitor:
    """
    Runtime monitor that verifies if the execution of QuantumPaxos
    conforms to the TLA+ specification.
    """
    def __init__(self, spec_path: str):
        self.spec_path = spec_path
        self.history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("TLA.Monitor")
        self.current_ballot = 0
        self.current_slot = 0

    def log_transition(self, action: str, node_id: str, old_state: dict, new_state: dict):
        """Log a state transition and perform immediate safety checks"""
        entry = {
            "action": action,
            "node": node_id,
            "before": old_state,
            "after": new_state,
        }
        self.history.append(entry)

        # Immediate safety checks
        if not self._check_transition_safety(entry):
            self.logger.error(f"SAFETY VIOLATION detected in action {action} on node {node_id}")

    def _check_transition_safety(self, entry: dict) -> bool:
        action = entry["action"]
        before = entry["before"]
        after = entry["after"]

        # 1. Ballot must be monotonically increasing (or same)
        if after.get("ballot", 0) < before.get("ballot", 0):
            self.logger.error("Ballot non-monotonic!")
            return False

        # 2. Slot must be monotonically increasing (or same)
        if after.get("slot", 0) < before.get("slot", 0):
            self.logger.error("Slot non-monotonic!")
            return False

        # 3. Specifically for COMMIT, slot must increase
        if action == "COMMIT":
            if after.get("slot", 0) <= before.get("slot", 0):
                self.logger.error("COMMIT didn't advance slot!")
                return False

        return True

    def verify_agreement(self) -> bool:
        """Verify the Agreement property: no two nodes decide different values for the same slot"""
        commits = {} # slot -> (value_hash, node)
        for entry in self.history:
            if entry["action"] == "COMMIT":
                slot = entry["after"]["slot"]
                val = entry["after"].get("value")
                if slot in commits:
                    prev_val, prev_node = commits[slot]
                    if prev_val != val:
                        self.logger.error(f"AGREEMENT VIOLATION: Slot {slot} has conflicting values from {prev_node} and {entry['node']}")
                        return False
                commits[slot] = (val, entry["node"])
        return True
