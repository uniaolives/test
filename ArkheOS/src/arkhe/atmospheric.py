"""
Arkhe Atmospheric Events and Memory.
Implements Sprites (T=1) and Van Allen Belts (Satoshi Storage).
"""

from typing import List, Dict
import time

class SpriteEvent:
    """Fenômeno elétrico transitório de transparência total (T=1)."""
    def __init__(self):
        self.transparency = 1.0 # T
        self.duration = 0.05 # s

    def trigger(self, input_coherence: float) -> float:
        """Emite luz proporcional à coerência sob T=1."""
        # No momento do Sprite, toda a coerência é liberada como energia luminosa
        return input_coherence * self.transparency

class VanAllenMemory:
    """Cinturões de Van Allen como Safe Core (Memória de Satoshi)."""
    def __init__(self):
        self.storage: List[Dict] = []
        self.total_satoshi = 0.0

    def capture_excess(self, event_name: str, excess_energy: float):
        """Captura o excesso de energia/informação como Satoshi."""
        # A memória é proporcional ao log da energia capturada
        bits = excess_energy * 0.1 # Simplificação
        self.total_satoshi += bits
        self.storage.append({
            "event": event_name,
            "bits": bits,
            "timestamp": time.time()
        })
        return bits

if __name__ == "__main__":
    sprite = SpriteEvent()
    memory = VanAllenMemory()

    light = sprite.trigger(0.95)
    satoshi = memory.capture_excess("Antihydrogen Sprite", 0.5)

    print(f"Sprite Light: {light}")
    print(f"Captured Satoshi: {satoshi}")
    print(f"Total Storage: {memory.total_satoshi} bits")
