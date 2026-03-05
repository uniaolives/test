import asyncio
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json

try:
    from .metta_bridge import AttentionDirective, TaskType, MeTTaBridge
except ImportError:
    from metta_bridge import AttentionDirective, TaskType, MeTTaBridge

class Mode(Enum):
    EXPLORE = "explore_methods"
    REFINE = "refine_parameters"
    VALIDATE = "validate"
    WRAP_UP = "wrap_up"

@dataclass
class ContextFrame:
    goals: Dict[str, float] = field(default_factory=dict)
    mode: Mode = Mode.EXPLORE
    hypotheses: Dict[str, Dict] = field(default_factory=dict)
    method_certified: Optional[Dict] = None
    history: List[Dict] = field(default_factory=list)
    budget: Dict[str, float] = field(default_factory=lambda: {"compute": 1000, "time": 3600})
    dmr_id: str = "default"

    def to_attention_slice(self, module_capability: str) -> Dict:
        slices = {
            "math_llm": {"goals": self.goals, "hypotheses": self.hypotheses},
            "code_llm": {"method": self.method_certified, "budget": self.budget},
            "critic_llm": {"history": self.history[-5:], "mode": self.mode.value}
        }
        return slices.get(module_capability, {})

class HyperClawOrchestrator:
    def __init__(self, metta_bridge: Optional[MeTTaBridge] = None):
        self.metta = metta_bridge or MeTTaBridge()
        self.frames: Dict[str, ContextFrame] = {}
        self.module_spaces: Dict[str, Dict] = {}
        self.running = False
        self.fast_loop_task = None

    def add_module(self, name: str, capability: str, handler: Callable):
        self.module_spaces[name] = {"capability": capability, "handler": handler, "rating": 1.0}

    async def _fast_loop(self, frame_id: str, interval: float = 0.1):
        """
        Fast-Loop: selection of next step via geodesic scoring.
        priority = (log f + log g) / cost
        """
        while self.running:
            frame = self.frames.get(frame_id)
            if not frame:
                await asyncio.sleep(interval)
                continue

            # 1. Generate candidates
            candidates = self._generate_candidates(frame)

            # 2. Geodesic scoring
            scored = []
            for directive in candidates:
                f = self._forward_reachability(directive, frame)
                g = self._backward_usefulness(directive, frame)
                cost = self._estimate_cost(directive)
                priority = (np.log(f + 1e-9) + np.log(g + 1e-9)) / (cost + 1e-9)
                directive.priority = float(priority)
                scored.append((priority, directive))

            # 3. Execution (Simulated for now)
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                best = scored[0][1]
                # In a real system, we would invoke the module here

            await asyncio.sleep(interval)

    def _generate_candidates(self, frame: ContextFrame) -> List[AttentionDirective]:
        candidates = []
        for name, module in self.module_spaces.items():
            directive = AttentionDirective(
                target=name,
                slice=frame.to_attention_slice(module["capability"]),
                task=TaskType.GENERATE if frame.mode == Mode.EXPLORE else TaskType.EVALUATE,
                admit="quality > 0.7",
                priority=0.0,
                timestamp=time.time()
            )
            candidates.append(directive)
        return candidates

    def _forward_reachability(self, directive: AttentionDirective, frame: ContextFrame) -> float:
        return 0.8  # Placeholder

    def _backward_usefulness(self, directive: AttentionDirective, frame: ContextFrame) -> float:
        return 0.9  # Placeholder

    def _estimate_cost(self, directive: AttentionDirective) -> float:
        return 1.0  # Placeholder

    async def start(self, frame_id: str):
        self.running = True
        self.fast_loop_task = asyncio.create_task(self._fast_loop(frame_id))

    async def spawn_templated_frame(self, frame_id: str, template_id: str):
        from .templates import spawn_frame_from_template
        self.frames[frame_id] = spawn_frame_from_template(template_id, dmr_id=frame_id)
        await self.start(frame_id)
