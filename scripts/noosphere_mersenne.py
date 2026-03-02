# scripts/noosphere_mersenne.py
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from papercoder_kernel.math.noosphere import MersenneNoosphere

def main():
    print("--- Mersenne Prime Search in the Noosphere ---")
    ns = MersenneNoosphere()

    limit = 31
    print(f"\nSearching for Mersenne primes up to p={limit}...")
    exponents = ns.find_mersenne_exponents(limit)
    print(f"Exponents found: {exponents}")

    print("\nCalculating Euclid-Euler Perfect Numbers:")
    perfeitos = ns.generate_perfect_numbers(exponents)
    for p, perf in perfeitos:
        print(f"  p={p:2d} -> M_p = {2**p-1:10d} -> Perfect = {perf}")

    print("\n🔮 A Noosfera está sondando...")
    proximos = ns.probe_noosphere_candidates(start=limit, count=10)
    print(f"Natural candidates (next prime exponents): {proximos}")

if __name__ == "__main__":
    main()
