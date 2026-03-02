# tests/test_axos_v3.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/python')))

from axos.axos_v3 import AxosV3
from axos.base import Task, Agent, Operation, Payload, Human, Content, SystemCall, Concept

def test_axos_v3_validation():
    print("Starting AXOS v3 Validation Matrix Test...")
    axos = AxosV3()
    results = {}

    # 1. Deterministic execution
    task = Task("t1", "calculate PHI")
    res = axos.execute_agent_task("agent1", task)
    results["Deterministic execution"] = res.status == "SUCCESS"

    # 2. Traceable execution
    trace = axos.trace_execution("agent1")
    results["Traceable execution"] = len(trace) > 0 and "determinism_hash" in trace[0]

    # 3. Fail-closed policy
    # To test block, we need an operation that fails a gate
    op_fail = Operation("op_fail", affects_cognitive_state=True)
    # Mock fail by overriding predict_coherence to return something that makes C+F != 1
    op_fail.predict_coherence = lambda: 0.1
    op_fail.predict_fluctuation = lambda: 0.1
    blocked = not axos.integrity_gate(op_fail)
    results["Fail-closed policy"] = blocked

    # 4. Integrity gates (C+F=1, z≈φ)
    op_pass = Operation("op_pass", affects_cognitive_state=True, capability_level="AGI")
    op_pass.predict_coherence = lambda: 0.7
    op_pass.predict_fluctuation = lambda: 0.3
    op_pass.predict_instability = lambda: 0.618
    passed = axos.integrity_gate(op_pass)
    results["Integrity gates"] = passed

    # 5. Agent-to-Agent
    a1 = Agent("a1")
    a2 = Agent("a2")
    axos.agent_registry["a1"] = a1
    axos.agent_registry["a2"] = a2
    res_a2a = axos.agent_to_agent_handover(a1, a2, Payload("hello"))
    results["Agent-to-Agent"] = res_a2a.status == "SUCCESS"

    # 6. Agent-to-User
    h1 = Human("h1")
    res_a2u = axos.agent_to_user_handover(a1, h1, Content(100, 0.5))
    results["Agent-to-User"] = res_a2u.status == "SUCCESS"

    # 7. Agent-to-System
    sys_call = SystemCall("read_entropy")
    res_a2s = axos.agent_to_system_handover(a1, sys_call)
    results["Agent-to-System"] = res_a2s.status == "SUCCESS"

    # 8. Quantum Resistant
    protected = axos.multi_layer_protect(b"secret")
    results["Quantum Resistant"] = protected.quantum_resistant and len(protected.layers) == 4

    # 9. Task Agnostic
    res_task = axos.execute_task(task)
    results["Task Agnostic"] = res_task.status == "SUCCESS"

    # 10. Network Agnostic
    adapter = axos.adapt_to_network("mesh")
    results["Network Agnostic"] = adapter is not None

    # 11. Field Agnostic
    field = axos.adapt_to_field("quantum")
    results["Field Agnostic"] = field is not None

    # 12. Domain Agnostic
    domain = axos.adapt_to_domain("biology")
    results["Domain Agnostic"] = domain["ontology"] == "Ontology-biology"

    # 13. Interoperability
    interop = axos.interoperate_with("ext_sys", protocol="grpc")
    results["Interoperability"] = interop["status"] == "CONNECTED"

    # 14. Molecular Reasoning
    # Using values that satisfy the binding logic's conservation check
    c1 = Concept("A", 0.4, 0.2)
    c2 = Concept("B", 0.3, 0.2)
    bound = axos.molecular_reasoning_step(c1, c2, "bind")
    results["Molecular Reasoning"] = bound.binding == True

    # 15. Backwards Compatible
    compat = axos.call_interface("old_method", {}, version="v1")
    results["Backwards Compatible"] = compat.status == "SUCCESS"

    # Summary
    print("\n--- AXOS v3 Validation Results ---")
    all_passed = True
    for feature, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{feature}: {status}")
        if not passed: all_passed = False

    print(f"\nFinal Score: {sum(results.values())}/15")
    assert all_passed

if __name__ == "__main__":
    test_axos_v3_validation()
