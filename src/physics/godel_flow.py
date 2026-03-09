# src/physics/godel_flow.py
# Gradient descent on Gödel-encoded stability

import torch
import torch.nn as nn
import numpy as np

class GodelFlow(nn.Module):
    """
    Flow in 1024D space where 'lowest point' = minimal Gödel complexity.
    The attractors are orbits with simple prime factorizations.
    """
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim
        # Learnable primes (or fixed first 1024 primes)
        self.primes = nn.Parameter(torch.tensor(self._first_primes(dim), dtype=torch.float),
                                   requires_grad=False)

    def _first_primes(self, n):
        primes = []
        num = 2
        while len(primes) < n:
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    break
            else:
                primes.append(num)
            num += 1
        return primes

    def godel_encode(self, x):
        """
        x: [batch, 1024] embedding
        Returns Gödel number as "complexity score"
        """
        # Active features = those above threshold
        # Using sigmoid as a differentiable approximation of the step function
        active = torch.sigmoid(100.0 * (torch.abs(x) - 0.1))

        # Gödel complexity = sum of log(primes) for active features
        # Lower = simpler = more stable
        complexity = (active * torch.log(self.primes)).sum(dim=-1)
        return complexity

    def forward(self, x):
        """
        'Water flowing' = gradient descent on Gödel complexity
        """
        # Find direction that minimizes complexity
        x_req = x.detach().clone().requires_grad_(True)
        complexity = self.godel_encode(x_req)

        # Flow toward lower complexity (stable ghosts)
        grad = torch.autograd.grad(complexity.sum(), x_req)[0]
        return x - 0.01 * grad  # Step "downhill"
