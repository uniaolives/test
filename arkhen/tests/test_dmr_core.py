import pytest
import time
import sys
import os

# Ensure we can import core_dmr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dmr')))

try:
    import core_dmr
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

@pytest.mark.skipif(not RUST_AVAILABLE, reason="core_dmr not available")
class TestDMR1:
    def setup_method(self):
        self.vk_ref = core_dmr.KatharosVector(0.5, 0.5, 0.5, 0.5)
        self.dmr = core_dmr.DigitalMemoryRing("test_agent_1", 0.5, 0.5, 0.5, 0.5, 100)

    def test_tkr_linear_growth(self):
        # 10 layers with stability
        for i in range(10):
            # simulate time passing
            # In our Rust impl, it checks dt between layers
            # But the first layer sets t_kr_seconds += 1 if no back
            self.dmr.grow_layer(0.5, 0.5, 0.5, 0.5)

        t_kr = self.dmr.measure_t_kr()
        assert t_kr > 0

    def test_tkr_reset_on_crisis(self):
        # Accumulate some t_kr
        self.dmr.grow_layer(0.5, 0.5, 0.5, 0.5)
        self.dmr.grow_layer(0.5, 0.5, 0.5, 0.5)
        assert self.dmr.measure_t_kr() > 0

        # Trigger crisis
        is_crisis = self.dmr.grow_layer(0.9, 0.1, 0.1, 0.1)
        assert is_crisis is True
        assert self.dmr.measure_t_kr() == 0

    def test_pattern_match(self):
        self.dmr.grow_layer(0.5, 0.5, 0.5, 0.5)
        self.dmr.grow_layer(0.6, 0.6, 0.6, 0.6)

        matches = self.dmr.heavy_pattern_match([0.5, 0.5, 0.5, 0.5])
        assert len(matches) >= 1
