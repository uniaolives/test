import json
import subprocess
import os

class DMR_OrbVM_Adapter:
    def __init__(self, orbvm_path="target/release/orbvm-cli"):
        # Try to find the binary relative to the repository root
        self.orbvm_path = orbvm_path

    def convert_dmr_to_orb(self, dmr_state):
        """
        dmr_state: dict with { 'q': coherence, 'dk': delta_k, 'msg': data }
        """
        # Simplification: maps DMR to OrbVM payload
        target_time = 1773414855 # Pi Day 2026 UTC

        cmd = [
            self.orbvm_path,
            "emit",
            "--data", dmr_state['msg'],
            "--target", str(target_time)
        ]

        try:
            # Check if binary exists
            if not os.path.exists(self.orbvm_path):
                 return f"Error: OrbVM binary not found at {self.orbvm_path}. Did you run 'cargo build --release' in the orbvm directory?"

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    adapter = DMR_OrbVM_Adapter()
    print(adapter.convert_dmr_to_orb({'msg': 'Pi Day 2026 Test', 'q': 0.92}))
