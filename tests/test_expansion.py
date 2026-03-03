# tests/test_expansion.py
import pytest
import asyncio
from cosmos.expansion import ExpansionOrchestrator, QuantumFusionPropulsion, QuantumMatterRevolution, QuantumSingularityPreparation
from ethical_optimizer import EthicalOptimizer

@pytest.mark.asyncio
async def test_milestone_deployment():
    fusion = QuantumFusionPropulsion()
    success = await fusion.deploy()
    assert success is True
    assert fusion.status == "OPERATIONAL"

@pytest.mark.asyncio
async def test_orchestrator_parallel():
    orchestrator = ExpansionOrchestrator()
    results = await orchestrator.run_parallel_deployment([1, 2, 5])
    assert all(results)
    assert len(results) == 3

def test_ethical_optimizer():
    optimizer = EthicalOptimizer()
    metrics = {
        "flourishing_score": 0.8,
        "efficiency_score": 0.7,
        "existential_risk": 0.05
    }
    assert optimizer.validate_action("Test Action", metrics) is True

    bad_metrics = {
        "flourishing_score": 0.1,
        "efficiency_score": 0.1,
        "existential_risk": 0.5
    }
    assert optimizer.validate_action("Bad Action", bad_metrics) is False
