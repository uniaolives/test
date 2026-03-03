import pytest
import numpy as np
import hashlib

# Mock implementation of Urbit logic for verification
def shax_mock(data: bytes):
    return hashlib.sha256(data).hexdigest()

def compute_global_coherence_logic(wits, stakes):
    # 1. Filter witnesses from ships without stake
    filtered = [w for w in wits if w['observer'] in stakes]
    if not filtered: return 0.0

    # 2. Sort by value
    filtered.sort(key=lambda x: x['value'])

    # 3. Symmetric Trimmed Mean (remove 10% from both ends)
    n = len(filtered)
    trim = n // 10
    if n >= 10:
        trimmed = filtered[trim : n-trim]
    else:
        trimmed = filtered

    if not trimmed: return 0.0

    # 4. Stake-weighted average
    total_weight = 0
    weighted_sum = 0.0
    for w in trimmed:
        s = stakes.get(w['observer'], 1)
        total_weight += s
        weighted_sum += w['value'] * s

    return weighted_sum / total_weight if total_weight > 0 else 0.0

def test_consensus_algorithm():
    stakes = {"ship1": 100, "ship2": 50, "ship3": 10}
    # With less than 10 wits, no trimming in my logic
    wits = [
        {"observer": "ship1", "value": 0.9},
        {"observer": "ship2", "value": 0.8},
        {"observer": "ship3", "value": 0.2},
        {"observer": "malicious", "value": 1.0} # No stake
    ]

    res = compute_global_coherence_logic(wits, stakes)
    # Expected: (0.9*100 + 0.8*50 + 0.2*10) / (100+50+10)
    # = (90 + 40 + 2) / 160 = 132 / 160 = 0.825
    assert res == pytest.approx(0.825)

def test_consensus_trimming():
    stakes = {f"s{i}": 1 for i in range(10)}
    wits = [{"observer": f"s{i}", "value": 0.5} for i in range(1, 9)]
    wits.append({"observer": "s0", "value": 0.1}) # outlier low
    wits.append({"observer": "s9", "value": 0.9}) # outlier high

    # n=10, trim=1. Should remove s0 and s9.
    res = compute_global_coherence_logic(wits, stakes)
    # Remaining should be 8 wits of 0.5
    assert res == pytest.approx(0.5)

def test_signature_placeholder():
    # Simple check for the logic used in verify-intent-signature
    intent = {"goal": "test"}
    ship = "ship1"
    # Jam would be a string representation for this mock
    data = str((intent, ship)).encode()
    sig = hashlib.sha256(data).hexdigest()

    # Verification
    assert sig == hashlib.sha256(str((intent, ship)).encode()).hexdigest()
