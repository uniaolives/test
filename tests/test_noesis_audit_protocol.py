
import pytest
import asyncio
from noesis_security_audit_protocol.core.quantum_integrity import QuantumIntegrityEngine
from noesis_security_audit_protocol.core.constitutional_guardian import ConstitutionalGuardian
from noesis_security_audit_protocol.core.multi_layer_scanner import MultiLayerSecurityScanner, LayerID
from noesis_security_audit_protocol.core.behavioral_forensics import BehavioralForensicsEngine
from noesis_security_audit_protocol.core.adversarial_resilience import AdversarialResilienceFramework

@pytest.mark.asyncio
async def test_protocol_integration():
    # 1. Initialize Engines
    quantum_engine = QuantumIntegrityEngine()
    guardian = ConstitutionalGuardian(quantum_engine)
    scanner = MultiLayerSecurityScanner(quantum_engine, guardian)
    forensics = BehavioralForensicsEngine()
    resilience = AdversarialResilienceFramework(quantum_engine, guardian)

    # 2. Test Quantum Initialization
    await quantum_engine.initialize_entanglement_network()
    assert len(quantum_engine.entanglement_registry) == 8

    # 3. Test Audit Record Creation
    payload = {"action": "test_operation", "status": "success"}
    audit = await quantum_engine.create_audit_record(
        layer="physical",
        action_payload=payload,
        criticality=0.5
    )
    assert audit.layer == "physical"
    assert len(quantum_engine.audit_chain) == 1

    # 4. Test Constitutional Assessment
    action = {"type": "resource_allocation", "amount": 1000}
    assessment = await guardian.assess_action(
        action=action,
        context={"environment": "production"},
        proposed_by="OPERATIONAL_AGENT"
    )
    assert assessment.overall_alignment > 0
    assert assessment.risk_classification is not None

    # 5. Test Multi-layer Scan
    scan_result = await scanner._execute_layer_scan(LayerID.INFRASTRUCTURE)
    assert scan_result.integrity_score >= 0.0
    assert scan_result.layer == LayerID.INFRASTRUCTURE

    print("Integration test passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_protocol_integration())
