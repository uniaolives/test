import numpy as np
import torch
import torch.nn as nn
from mmgroup import MM, MMV, mmv_scalprod

class GriessLayer(nn.Module):
    """
    Griess Layer: A high-dimensional representation layer using the
    Monster group's Griess algebra (via mmgroup).

    It maps informational bits to the 196,884-dimensional representation
    and extracts invariants under the action of the Monster.
    """
    def __init__(self, characteristic=255):
        super().__init__()
        self.p = characteristic
        self.V = MMV(self.p)
        self.dim = 196884

        # Trainable parameters that act as "weights" in the Monster Group
        # Represented as an element of the Monster Group
        self.monster_element = MM() # Identity element

    def embed_bits(self, bits: str) -> any:
        """
        Embeds an 85-bit string into a vector in the Griess algebra.
        We map bits to indices of the representation.
        """
        if len(bits) != 85:
            # Pad or truncate if not 85 bits
            bits = bits.ljust(85, '0')[:85]

        # Strategy: Use bits to define tags for MMVector
        # MMVector(p, tag, i0, i1)
        v = self.V() # Null vector

        # Map bits to various components of the representation
        # Indices for X, Y, Z tags: i0 in [0, 2047], i1 in [0, 23]
        # Indices for tag A: i0 in [0, 23], i1 in [0, 23]

        v += self.V('X', int(bits[0:11], 2) % 2048, int(bits[11:16], 2) % 24)
        v += self.V('Z', int(bits[24:35], 2) % 2048, int(bits[35:40], 2) % 24)
        v += self.V('Y', int(bits[48:59], 2) % 2048, int(bits[59:64], 2) % 24)
        v += self.V('A', int(bits[72:77], 2) % 24, int(bits[77:82], 2) % 24)

        return v

    def calculate_invariant(self, v) -> float:
        """
        Calculates a scalar invariant (quadratic form) of the vector.
        We use the scalar product of the vector with its transformed version.
        """
        # Apply the trainable Monster element to the vector
        v_transformed = v * self.monster_element

        # Return the scalar product (invariant under group action if monster_element is from M)
        return float(mmv_scalprod(v, v_transformed))

    def forward(self, bits_batch: list[str]) -> torch.Tensor:
        """
        Forward pass for a batch of bit strings.
        Returns a tensor of invariants.
        """
        invariants = []
        for bits in bits_batch:
            v = self.embed_bits(bits)
            inv = self.calculate_invariant(v)
            invariants.append(inv)

        return torch.tensor(invariants, dtype=torch.float32).unsqueeze(-1)

    def update_monster_element(self, *args, **kwargs):
        """
        Updates the internal Monster group element.
        """
        self.monster_element = MM(*args, **kwargs)
