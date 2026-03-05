import sys
import os

# Path to the compiled Rust library
RUST_LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../target/release"))
sys.path.append(RUST_LIB_PATH)

try:
    import dmr_bridge
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print(f"[WARNING] Rust dmr_bridge not found at {RUST_LIB_PATH}. Using mock.")

class MockStateLayer:
    def __init__(self, timestamp, bio, aff, soc, cog):
        self.timestamp = timestamp
        self.bio = bio
        self.aff = aff
        self.soc = soc
        self.cog = cog

class MockDigitalMemoryRing:
    def __init__(self, agent_id, bio, aff, soc, cog):
        self.id = agent_id
        self.vk_ref = {"bio": bio, "aff": aff, "soc": soc, "cog": cog}
        self.layers = []

    def grow_layer(self, bio, aff, soc, cog, q):
        import time
        self.layers.append(MockStateLayer(
            int(time.time()),
            bio, aff, soc, cog
        ))

    def measure_t_kr(self):
        return len(self.layers) * 3600

    def reconstruct_trajectory(self):
        return self.layers

    def get_stats(self):
        import json
        return json.dumps({
            "id": self.id,
            "layer_count": len(self.layers),
            "t_kr": self.measure_t_kr(),
            "vk_ref_bio": self.vk_ref["bio"],
            "vk_ref_aff": self.vk_ref["aff"],
            "vk_ref_soc": self.vk_ref["soc"],
            "vk_ref_cog": self.vk_ref["cog"],
        })

def get_dmr_instance(agent_id: str):
    if RUST_AVAILABLE:
        return dmr_bridge.PyDigitalMemoryRing(agent_id, 0.5, 0.5, 0.5, 0.5)
    else:
        return MockDigitalMemoryRing(agent_id, 0.5, 0.5, 0.5, 0.5)
