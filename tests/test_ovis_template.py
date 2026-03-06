import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.app.hyperclaw.loops import HyperClawOrchestrator, Mode

@pytest.mark.asyncio
async def test_ovis_template_spawning():
    orch = HyperClawOrchestrator()
    await orch.spawn_templated_frame("ovis_frame", "ruminant_genetics_ovis")

    assert "ovis_frame" in orch.frames
    frame = orch.frames["ovis_frame"]
    assert frame.goals["karyotype_stability_2n54"] == 1.0
    assert frame.goals["rumen_metabolic_efficiency"] == 0.85
    assert frame.mode == Mode.EXPLORE
    assert frame.budget["compute"] == 4500
    assert orch.running == True
    orch.running = False

if __name__ == "__main__":
    pytest.main([__file__])
