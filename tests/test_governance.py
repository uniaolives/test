import pytest
import asyncio
from papercoder_kernel.governance.core import ASIGovernanceCore, GovernanceState
from papercoder_kernel.governance.ethics import ArkheASI_Ethics
from papercoder_kernel.governance.risk_model import ASIRiskModel

class MockASI:
    def __init__(self, compliance_map):
        self.compliance_map = compliance_map
    def check_principle(self, principle, config):
        return self.compliance_map.get(principle, True)

def test_ethics_compliance():
    # Test compliant system
    asi_ok = MockASI({})
    result_ok = ArkheASI_Ethics.verify_compliance(asi_ok)
    assert result_ok['compliant'] is True
    assert result_ok['score'] == 1.0

    # Test non-compliant system
    asi_fail = MockASI({'TRANSPARENCY_RADICAL': False, 'COHERENCE_VERIFICATION': False})
    result_fail = ArkheASI_Ethics.verify_compliance(asi_fail)
    assert result_fail['compliant'] is False
    assert result_fail['score'] < 0.95
    assert 'TRANSPARENCY_RADICAL' in result_fail['violations']

@pytest.mark.asyncio
async def test_governance_core_monitoring():
    # Use a flag to check if kill switch was called
    kill_switch_called = False
    def mock_kill_switch(state):
        nonlocal kill_switch_called
        kill_switch_called = True

    gov = ASIGovernanceCore(kill_switch_callback=mock_kill_switch)

    # Normal state
    state_ok = {
        'phi': 0.005,
        'coherence': 0.9,
        'entropy_local': 0.1,
        'entropy_global': 100.1
    }
    result = await gov.governance_cycle(state_ok)
    assert result.compliant is True
    assert kill_switch_called is False

    # Critical state (low coherence)
    state_crit = {
        'phi': 0.005,
        'coherence': 0.4, # Below 0.5 threshold
        'entropy_local': 0.1,
        'entropy_global': 100.2
    }
    result_crit = await gov.governance_cycle(state_crit)
    assert result_crit.compliant is False
    assert 'COHERENCE_CRITICAL' in result_crit.violations
    assert kill_switch_called is True

def test_risk_model():
    risk_model = ASIRiskModel()

    # High coherence, low Phi -> Low risk
    state_safe = {'coherence': 0.95, 'phi': 0.005}
    res_safe = risk_model.calculate_risk(state_safe)
    assert res_safe['total_risk'] < 0.2

    # Low coherence, high Phi -> High risk
    state_danger = {'coherence': 0.1, 'phi': 0.09}
    res_danger = risk_model.calculate_risk(state_danger)
    assert res_danger['total_risk'] > 0.5
