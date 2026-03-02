import pytest
import asyncio
from merkabah.quantum.qhttp import QHTTPClient, QHTTPRequest, QHTTPMethod, QHTTPStatusCode
import numpy as np

@pytest.mark.asyncio
async def test_qhttp_methods_compliance():
    client = QHTTPClient()

    # Test SUPERPOSE
    req_super = QHTTPRequest(method=QHTTPMethod.SUPERPOSE, target="quantum://moduli/explore")
    resp_super = await client.request(req_super)
    assert resp_super.status_code == QHTTPStatusCode.SUPERPOSED
    assert resp_super.coherence > 0.9

    # Test ENTANGLE
    req_ent = QHTTPRequest(method=QHTTPMethod.ENTANGLE, target="quantum://generator/entangle")
    resp_ent = await client.request(req_ent)
    assert resp_ent.status_code == QHTTPStatusCode.ENTANGLED
    assert resp_ent.entanglement_id is not None

    # Test DECOHERE
    req_deco = QHTTPRequest(method=QHTTPMethod.DECOHERE, target="quantum://entity/critical")
    resp_deco = await client.request(req_deco)
    assert resp_deco.status_code == QHTTPStatusCode.OK
    assert resp_deco.coherence == 0.0

def test_circuit_generation():
    # Test for all methods circuit generation
    for method in QHTTPMethod:
        req = QHTTPRequest(method=method, target="quantum://test#0-4")
        circuit = req.to_quantum_circuit()
        if circuit:
            assert circuit.name == f"qhttp_{method.name}"
            assert len(circuit.qubits) == 4

def test_header_encoding():
    req1 = QHTTPRequest(method=QHTTPMethod.MEASURE, target="quantum://test", headers={"X-Key": "Value1"})
    req2 = QHTTPRequest(method=QHTTPMethod.MEASURE, target="quantum://test", headers={"X-Key": "Value2"})

    c1 = req1.to_quantum_circuit()
    c2 = req2.to_quantum_circuit()

    if c1 and c2:
        # Circuits should be different due to different headers
        assert str(c1) != str(c2)
