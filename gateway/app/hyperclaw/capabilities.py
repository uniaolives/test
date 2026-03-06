from typing import Set, Dict, List
from dataclasses import dataclass

@dataclass
class CapabilityContract:
    frame_read: Set[str]
    frame_write: Set[str]
    compute_budget: float
    data_access: Set[str]
    network_allowlist: Set[str]

    def check_access(self, attempted_field: str, direction: str) -> bool:
        if direction == "read":
            return attempted_field in self.frame_read
        elif direction == "write":
            return attempted_field in self.frame_write
        return False

class CapabilityEnforcer:
    def __init__(self):
        self.contracts: Dict[str, CapabilityContract] = {}

    def register(self, module_name: str, contract: CapabilityContract):
        self.contracts[module_name] = contract
