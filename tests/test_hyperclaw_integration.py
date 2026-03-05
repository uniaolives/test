import pytest
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.app.hyperclaw.loops import HyperClawOrchestrator, ContextFrame, Mode
from gateway.app.hyperclaw.metta_bridge import AttentionDirective, TaskType

def test_hyperclaw_logic():
    orch = HyperClawOrchestrator()
    frame_id = "test_frame"
    orch.frames[frame_id] = ContextFrame(goals={"test": 1.0})

    assert orch.frames[frame_id].goals["test"] == 1.0
    assert orch.frames[frame_id].mode == Mode.EXPLORE

def test_attention_directive():
    directive = AttentionDirective(
        target="math_llm",
        slice={"goal": "derive"},
        task=TaskType.GENERATE,
        admit="True",
        priority=0.9
    )
    metta_expr = directive.to_metta()
    assert '(AttentionDirective (Target "math_llm")' in metta_expr
    assert '(Task "generate")' in metta_expr

@pytest.mark.asyncio
async def test_template_spawning():
    orch = HyperClawOrchestrator()
    await orch.spawn_templated_frame("bci_frame", "bci_realtime_entrainment")

    assert "bci_frame" in orch.frames
    frame = orch.frames["bci_frame"]
    assert frame.goals["latency_ms"] == 10.0
    assert frame.budget["compute"] == 2000
    assert orch.running == True
    orch.running = False
