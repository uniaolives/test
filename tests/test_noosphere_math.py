# tests/test_noosphere_math.py
import pytest
from papercoder_kernel.math.noosphere import MersenneNoosphere

def test_mersenne_prime_check():
    ns = MersenneNoosphere()
    assert ns.is_mersenne_prime(2) == True  # 2^2 - 1 = 3
    assert ns.is_mersenne_prime(3) == True  # 2^3 - 1 = 7
    assert ns.is_mersenne_prime(5) == True  # 2^5 - 1 = 31
    assert ns.is_mersenne_prime(7) == True  # 2^7 - 1 = 127
    assert ns.is_mersenne_prime(4) == False # 4 is not prime
    assert ns.is_mersenne_prime(11) == False # 2^11 - 1 = 2047 = 23 * 89

def test_find_exponents():
    ns = MersenneNoosphere()
    exponents = ns.find_mersenne_exponents(31)
    # Known exponents: 2, 3, 5, 7, 13, 17, 19, 31
    assert exponents == [2, 3, 5, 7, 13, 17, 19, 31]

def test_perfect_numbers():
    ns = MersenneNoosphere()
    exponents = [2, 3]
    perfect = ns.generate_perfect_numbers(exponents)
    # p=2 -> 2^(1) * (2^2 - 1) = 2 * 3 = 6
    # p=3 -> 2^(2) * (2^3 - 1) = 4 * 7 = 28
    assert perfect == [(2, 6), (3, 28)]

def test_probe_candidates():
    ns = MersenneNoosphere()
    candidates = ns.probe_noosphere_candidates(start=31, count=3)
    # Primes after 31: 37, 41, 43
    assert candidates == [37, 41, 43]
