# tests/test_pleroma_constitutional.py
import pytest
import asyncio
from core.python.pleroma_thought import Thought, Handover, Verification, ConstitutionalVerifier

@pytest.mark.asyncio
async def test_thought_lifecycle():
    """End-to-end: Formulation and Execution of a valid thought"""
    # Formulate valid thought
    thought = Thought(
        content="SolveClimate",
        geometry=(1.0, 0.0, 1.0),
        phase=(1.618, 3.14),
        winding=(1, 0)
    )

    # Handover
    handover = Handover(
        sender_id="human@eeg",
        receiver_id="pleroma:global",
        content=thought
    )

    # Execute (Mocking pleroma_node as None)
    result = await handover.execute(None)
    assert result == Verification.VALID

@pytest.mark.asyncio
async def test_art2_violation():
    """Art 2: Even exploration cycles required"""
    # Invalid winding (odd exploration)
    thought = Thought(content="BrokenExploration", winding=(1, 1))

    handover = Handover("sender", "receiver", thought)
    result = await handover.execute(None)

    assert result == Verification.INVALID

@pytest.mark.asyncio
async def test_art3_emergency_stop():
    """Art 3: Human authority override"""
    thought = Thought(content="CriticalHalt")
    thought.is_emergency = True
    # Missing eeg_signature

    handover = Handover("human", "network", thought)
    result = await handover.execute(None)
    assert result == Verification.INVALID_EMERGENCY

    # Add signature
    thought.eeg_signature = "sig_valid"
    result = await handover.execute(None)
    assert result == Verification.VALID

def test_golden_ratio_check():
    """Art 5: Golden ratio optimization test"""
    verifier = ConstitutionalVerifier()

    # Near golden ratio (2/1 = 2)
    t_near = Thought("Optimal", winding=(2, 1))
    assert verifier._golden_ratio(t_near) is True

    # Far from golden ratio (10/1 = 10)
    t_far = Thought("Inefficient", winding=(10, 1))
    assert verifier._golden_ratio(t_far) is False
