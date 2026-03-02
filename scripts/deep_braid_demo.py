# scripts/deep_braid_demo.py
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from papercoder_kernel.math.braid import DeepBraidArchitecture

def main():
    # Instanciar para p=61
    braid_p61 = DeepBraidArchitecture(61)
    word = braid_p61.generate_braid_word()
    invariants = braid_p61.compute_invariants()
    stable = braid_p61.stability_check()

    print(f"Trança profunda para p={braid_p61.p}")
    print(f"Mersenne: 2^{braid_p61.p} - 1")
    print(f"Perfect: 2^{braid_p61.p-1} * (2^{braid_p61.p} - 1)")
    print(f"Palavra (primeiros 20): {word[:20]}...")
    print(f"Invariantes: {invariants}")
    print(f"Estabilidade: {'OK' if stable else 'Instável'}")

if __name__ == "__main__":
    main()
