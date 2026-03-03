import torch
import pytest
import os
import sys
import numpy as np

# Ensure the local merkabah modules are in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/python")))

from merkabah.agi.griess_layer import GriessLayer

def test_griess_embedding():
    griess = GriessLayer(characteristic=255)
    bits = "1" * 85
    v = griess.embed_bits(bits)

    # Check that it's a valid MMVector
    assert v.p == 255

def test_griess_invariant():
    griess = GriessLayer(characteristic=255)
    bits = "1" * 85
    v = griess.embed_bits(bits)
    inv = griess.calculate_invariant(v)

    assert isinstance(inv, float)

def test_griess_forward():
    griess = GriessLayer(characteristic=255)
    batch = ["1" * 85, "0" * 85]
    output = griess(batch)

    assert output.shape == (2, 1)
    assert isinstance(output, torch.Tensor)

def test_monster_transformation():
    # Use a large characteristic to avoid collisions in small modulo
    griess = GriessLayer(characteristic=255)
    bits = "1" * 85
    v = griess.embed_bits(bits)

    inv_identity = griess.calculate_invariant(v)

    # Update monster element to a non-identity element (e.g., a generator 'd')
    # Using 'p' which is a permutation generator, usually has more effect
    griess.update_monster_element('p', 12345)
    inv_transformed = griess.calculate_invariant(v)

    # The invariant scalprod(v, v*g) should generally change if g != I
    assert inv_identity != inv_transformed
