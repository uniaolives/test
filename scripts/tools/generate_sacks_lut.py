#!/usr/bin/env python3
# tools/generate_sacks_lut.py
import numpy as np
import sympy
from typing import List

def generate_sacks_lut(max_primes: int = 1024, output_file: str = "sacks_lut.hex"):
    """
    Gera arquivo de inicialização BRAM para navegação da espiral de Sacks.
    Cada entrada: 64 bits = {prime[32], theta[16], neighbors[16]}
    """
    primes = list(sympy.primerange(2, 10000))[:max_primes]
    entries = []

    for idx, p in enumerate(primes):
        r = np.sqrt(idx + 1)
        theta = (2 * np.pi * r) % (2 * np.pi)
        theta_fixed = int((theta / (2 * np.pi)) * 65535)

        # Simple neighbor mapping for the prototype
        next_idx = (idx + 1) % max_primes
        word = (p << 32) | (theta_fixed << 16) | next_idx
        entries.append(f"{word:016X}")

    with open(output_file, 'w') as f:
        f.write('\n'.join(entries))

    print(f"🜏 {len(entries)} Sacks entries generated in {output_file}")

if __name__ == "__main__":
    import os
    os.makedirs("scripts/tools", exist_ok=True)
    generate_sacks_lut(output_file="sacks_lut.hex")
