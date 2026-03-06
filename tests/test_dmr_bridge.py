import sys
import os

# Ensure the compiled Rust library is in the path
RUST_LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../target/release"))
sys.path.append(RUST_LIB_PATH)

import pytest
try:
    import dmr_bridge
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

@pytest.mark.skipif(not RUST_AVAILABLE, reason="dmr_bridge not compiled")
def test_dmr_core():
    """Test the Rust core via the PyO3 bridge"""
    # KatharosVector::new(bio, aff, soc, cog)
    # PyDigitalMemoryRing::new(id, bio, aff, soc, cog)
    ring = dmr_bridge.PyDigitalMemoryRing("test-agent", 0.5, 0.5, 0.5, 0.5)

    # Grow 10 stable layers
    for _ in range(10):
        ring.grow_layer(0.52, 0.48, 0.51, 0.49, 0.95)

    t_kr = ring.measure_t_kr()
    assert t_kr > 0

    trajectory = ring.reconstruct_trajectory()
    assert len(trajectory) == 10
    assert trajectory[0].bio == 0.52
