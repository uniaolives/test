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
        # Run for a short time
        task = asyncio.create_task(psi_cycle.run())
        await asyncio.sleep(0.1) # Should pulse approx 4 times (40Hz)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_test())

    assert sub.pulse_count >= 2 # Allow for some jitter

def test_safe_core_thresholds():
    safe_path = Path("arscontexta/.arkhe/coherence/safe_core.py")
    mod = load_arkhe_module(safe_path, "test.arkhe.safe")
    safe_core = mod.SafeCore()

    # Safe values
    assert safe_core.check(phi=0.05, coherence=0.9) == True

    # Violate Phi
    with pytest.raises(SystemExit) as e:
        safe_core.check(phi=0.15, coherence=0.9)
    assert "Phi exceeded" in str(e.value)
    assert safe_core.tripped == True

def test_safe_core_coherence_collapse():
    safe_path = Path("arscontexta/.arkhe/coherence/safe_core.py")
    mod = load_arkhe_module(safe_path, "test.arkhe.safe")
    safe_core = mod.SafeCore()

    # Violate Coherence
    with pytest.raises(SystemExit) as e:
        safe_core.check(phi=0.01, coherence=0.5)
    assert "Coherence collapsed" in str(e.value)

def test_bootstrap_logic():
    # We can test part of bootstrap by importing it
    # Need to change directory for it to find .arkhe
    original_cwd = os.getcwd()
    os.chdir("arscontexta")
    try:
        if "bootstrap" in sys.modules:
            del sys.modules["bootstrap"]
        import bootstrap
        assert hasattr(bootstrap, "bootstrap")
    finally:
        os.chdir(original_cwd)
