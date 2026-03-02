# src/papercoder_kernel/math/noosphere.py
from sympy.ntheory import isprime
from typing import List, Tuple

class MersenneNoosphere:
    """
    Search for Mersenne primes and perfect numbers in the Noosphere.
    M_p = 2^p - 1
    Perfect = 2^(p-1) * (2^p - 1)
    """

    @staticmethod
    def is_mersenne_prime(p: int) -> bool:
        """Checks if 2^p - 1 is prime."""
        if not isprime(p):
            return False
        return isprime(2**p - 1)

    def find_mersenne_exponents(self, limit: int = 50) -> List[int]:
        """Finds exponents p up to limit such that 2^p - 1 is prime."""
        exponents = []
        for p in range(2, limit + 1):
            if self.is_mersenne_prime(p):
                exponents.append(p)
        return exponents

    def generate_perfect_numbers(self, exponents: List[int]) -> List[Tuple[int, int]]:
        """Generates Euclid-Euler perfect numbers from Mersenne exponents."""
        return [(p, 2**(p-1) * (2**p - 1)) for p in exponents]

    def probe_noosphere_candidates(self, start: int = 31, count: int = 10) -> List[int]:
        """
        Simulates the Noosphere probing for the next natural candidates (prime exponents).
        """
        candidates = []
        p = start + 1
        while len(candidates) < count:
            if isprime(p):
                candidates.append(p)
            p += 1
        return candidates
