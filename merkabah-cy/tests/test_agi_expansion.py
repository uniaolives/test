import torch
import pytest
import os
import sys

# Ensure the local merkabah modules are in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/python")))

from merkabah.agi.recursive_expansion import RecursiveRankTensor, AutoreferentialLoss

def test_recursive_expansion_trigger():
    initial_dim = 16
    # Low threshold to trigger expansion easily
    tensor = RecursiveRankTensor(initial_dim=initial_dim, entropy_threshold=0.01)

    # High entropy input
    x = torch.randn(1, initial_dim)
    output = tensor(x)

    assert tensor.current_dim > initial_dim
    assert output.shape[-1] == tensor.current_dim

def test_recursive_expansion_stability():
    initial_dim = 16
    # High threshold to prevent expansion
    tensor = RecursiveRankTensor(initial_dim=initial_dim, entropy_threshold=100.0)

    x = torch.zeros(1, initial_dim)
    output = tensor(x)

    assert tensor.current_dim == initial_dim
    assert output.shape[-1] == initial_dim

def test_autoreferential_loss():
    criterion = AutoreferentialLoss(growth_weight=0.5)
    pred = torch.randn(1, 10)
    target = torch.randn(1, 10)
    rank = 100

    loss = criterion(pred, target, rank)
    assert loss > 0

    # Check that higher rank decreases loss (reward)
    loss_higher_rank = criterion(pred, target, 1000)
    assert loss_higher_rank < loss
