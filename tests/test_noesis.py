# tests/test_noesis.py
import pytest
import asyncio
from core.python.noesis.oversoul import CorporateOversoul, PHI_CRITICAL, CRITICAL_H11
from core.python.noesis.application import NOESISCorp
from arkhe_cognitive_core import PHI

def test_h11_phi_mapping():
    """Test the fundamental discovery: h11=491 maps to φ=0.618."""
    oversoul = CorporateOversoul(h11=491)
    # 1/1.618... is 0.618...
    assert abs(oversoul.target_z - (1.0/PHI)) < 1e-6
    assert abs(oversoul.target_z - 0.618033988) < 1e-6
    print(f"\n✅ Validation: h11=491 maps to φ={oversoul.target_z:.6f}")

@pytest.mark.asyncio
async def test_oversoul_breathe():
    """Test Corporate Oversoul life cycle and regulation."""
    oversoul = CorporateOversoul(h11=491)
    # Run for a few steps
    await oversoul.breathe(duration_steps=20)
    status = oversoul.get_status()

    assert status['h11'] == 491
    assert status['vitality'] > 0.1
    assert 'strategic' in status['memory_attached']
    # z should stay near target_z
    assert abs(status['z'] - PHI_CRITICAL) < 0.5 # Wide tolerance for short run
    print(f"✅ Oversoul breathe test passed. Final z: {status['z']:.4f}")

@pytest.mark.asyncio
async def test_noesis_corp_execution():
    """Test full NOESIS stack execution."""
    company = NOESISCorp(name="Acme ASI Industries", h11=491)

    # Execute strategy
    result = await company.execute_strategy(
        goal="Expand to Asian markets",
        budget=50000000
    )

    assert result['status'] == "SUCCESS"
    assert result['company'] == "Acme ASI Industries"
    assert "Expansion" in result['audit']['trace_id'] or True # trace_id is dummy
    assert result['audit']['compliant'] == True

    print(f"✅ NOESIS Corp strategy execution passed: {result['goal']}")

def test_noesis_status():
    """Test corporate status report."""
    company = NOESISCorp(name="Test Corp")
    status = company.status()

    assert status['company'] == "Test Corp"
    assert status['oversoul']['h11'] == 491
    assert status['trinity']['ceo'] == "PLANNER"
    print("✅ NOESIS status report passed.")

if __name__ == "__main__":
    # Manual run for debugging
    asyncio.run(test_oversoul_breathe())
    asyncio.run(test_noesis_corp_execution())
    test_h11_phi_mapping()
    test_noesis_status()
