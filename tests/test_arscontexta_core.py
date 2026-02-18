import pytest
import asyncio
import sys
import os
from pathlib import Path
import importlib.util

# Add arscontexta to path to simulate being inside it
sys.path.append(str(Path("arscontexta").absolute()))

def load_arkhe_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None:
        raise ImportError(f"Could not load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_psi_cycle_pulses():
    psi_path = Path("arscontexta/.arkhe/Ψ/pulse_40hz.py")
    mod = load_arkhe_module(psi_path, "test.arkhe.psi")
    psi_cycle = mod.PsiCycle()

    class Subscriber:
        def __init__(self):
            self.pulse_count = 0
        def on_psi_pulse(self, phase):
            self.pulse_count += 1

    sub = Subscriber()
    psi_cycle.subscribe(sub)

    async def run_test():
        task = asyncio.create_task(psi_cycle.run())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_test())
    assert sub.pulse_count >= 2

def test_safe_core_thresholds():
    safe_path = Path("arscontexta/.arkhe/coherence/safe_core.py")
    mod = load_arkhe_module(safe_path, "test.arkhe.safe")
    safe_core = mod.SafeCore()
    assert safe_core.check(phi=0.05, coherence=0.9) == True
    with pytest.raises(SystemExit) as e:
        safe_core.check(phi=0.15, coherence=0.9)
    assert "Phi exceeded" in str(e.value)

def test_handover_quantum_classical():
    safe_path = Path("arscontexta/.arkhe/coherence/safe_core.py")
    safe_mod = load_arkhe_module(safe_path, "test.arkhe.safe.handover")
    safe_core = safe_mod.SafeCore()

    handover_path = Path("arscontexta/.arkhe/handover/quantum_classical.py")
    handover_mod = load_arkhe_module(handover_path, "test.arkhe.handover.qc")
    qc_handover = handover_mod.QuantumToClassicalHandover(safe_core)

    result = qc_handover.execute({"psi": "dummy"})
    assert result["state"] == "observed"
    assert qc_handover.latency_ms < 25

def test_ledger_verify_script():
    verify_path = Path("arscontexta/.arkhe/ledger/verify.py")
    verify_mod = load_arkhe_module(verify_path, "test.arkhe.ledger.verify")
    # Should work even with empty chain for now
    assert verify_mod.verify_chain(Path("arscontexta/.arkhe/ledger/chain")) == True

def test_bootstrap_logic():
    original_cwd = os.getcwd()
    os.chdir("arscontexta")
    try:
        if "bootstrap" in sys.modules:
            del sys.modules["bootstrap"]
        import bootstrap
        assert hasattr(bootstrap, "bootstrap")
    finally:
        os.chdir(original_cwd)
