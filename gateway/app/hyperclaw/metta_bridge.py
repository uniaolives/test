import json
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

class TaskType(Enum):
    GENERATE = "generate"
    CRITIQUE = "critique"
    REVISE = "revise"
    EXECUTE = "execute"
    EVALUATE = "evaluate"

@dataclass
class AttentionDirective:
    target: str
    slice: Dict[str, Any]
    task: TaskType
    admit: str
    priority: float
    timestamp: float = 0.0

    def to_metta(self) -> str:
        return f'(AttentionDirective (Target "{self.target}") (Slice {json.dumps(self.slice)}) (Task "{self.task.value}") (Admit "{self.admit}") (Priority {self.priority}) (Timestamp {self.timestamp}))'

    def cid(self) -> str:
        content = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class MeTTaBridge:
    def __init__(self, metta_bin: str = "metta"):
        self.metta_bin = metta_bin

    def execute(self, expr: str, context: Optional[Dict] = None) -> List[Any]:
        return self._python_fallback(expr, context)

    def _python_fallback(self, expr: str, context: Optional[Dict]) -> List[Any]:
        if "AttentionDirective" in expr and context:
            return [{"matched": True, "admit_eval": True}]
        return []
