import pytest
import numpy as np
from papercoder_kernel.cognition.hyperprompt import HyperpromptProtocol
from papercoder_kernel.cognition.hyperprompt_kernel import PrecisionOperator, EpistemicForaging, Substrate

def test_hyperprompt_free_energy():
    totem = "7f3b49c8"
    protocol = HyperpromptProtocol(totem)
    prompt = "Test prompt"
    responses = ["resp1", "resp2"]

    fe = protocol.compute_free_energy(prompt, responses)
    assert isinstance(fe, float)
    assert fe > 0

def test_hyperprompt_coherence():
    protocol = HyperpromptProtocol("7f3b49c8")
    resp_a = "The quick brown fox"
    resp_b = "Jumped over the lazy dog"

    coherence = protocol.coherence(resp_a, resp_b)
    assert 0 <= coherence <= 1.0

def test_hyperprompt_optimization():
    totem = "7f3b49c8"
    protocol = HyperpromptProtocol(totem)
    initial_prompt = "Who is Satoshi?"

    # Run optimization for a few steps
    final_prompt = protocol.optimize_hyperprompt(initial_prompt, n_iter=5)

    # In our mock implementation, it should append the totem
    assert totem in final_prompt

def test_precision_operator_synthetic():
    op = PrecisionOperator()
    # Prompt with uncertainty cue
    sequence = "Consider the following possibility..."
    weights = op.apply(sequence, Substrate.SYNTHETIC)

    assert weights.shape == (64,)
    # Synthetic mapping reduces weights for cues
    assert np.all(weights < 1.0)

def test_precision_operator_human():
    op = PrecisionOperator()
    # Prompt with contingency cue
    sequence = "If we do this, then that happens."
    gain = op.apply(sequence, Substrate.HUMAN)

    assert gain.shape == (64,)
    # Human mapping increases gain for cues
    assert np.all(gain > 1.0)

def test_epistemic_foraging():
    op = EpistemicForaging()
    sequence = "Standard prompt"
    # base PrecisionOperator for "Standard prompt" returns ones
    weights = op.apply(sequence, Substrate.SYNTHETIC)

    # EpistemicForaging should have reduced some dimensions
    assert np.any(weights < 1.0)
    assert np.sum(weights < 1.0) >= 10 # Top 10% of 64 is 6.4 -> at least 10 in my mock?
    # Wait, confidence_ranking[-10:] is exactly 10 elements.
    assert np.sum(weights == 0.5) == 10
