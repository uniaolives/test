# src/papercoder_kernel/math/braid.py
import numpy as np
from sympy import isprime

class DeepBraidArchitecture:
    """
    Arquitetura para sustentar a densidade de um número perfeito de Mersenne.
    Usa o primo p como número de fios.
    """

    def __init__(self, p: int):
        self.p = p
        self.mersenne = 2**p - 1
        self.perfect = 2**(p-1) * self.mersenne
        self.dim = p  # número de fios

    def generate_braid_word(self):
        """
        Gera a palavra da trança (sequência de geradores σ_i).
        Usa a expansão binária do número perfeito para determinar a sequência.
        """
        bits = bin(self.perfect)[2:]
        word = []
        for i, b in enumerate(bits):
            if b == '1':
                # adiciona um gerador baseado na posição
                # σ_i onde i está em [1, dim-1]
                g = (i % (self.dim - 1)) + 1
                word.append(f"σ_{g}")
        return word

    def compute_invariants(self):
        """
        Calcula simulacros de invariantes de Jones e HOMFLY-PT para a trança.
        Baseado na estrutura de Mersenne.
        """
        jones_poly = f"q^{self.p} - q^{self.p-2} + ..."  # representação simbólica
        homfly = f"α^{self.mersenne} + β^{self.perfect}"
        return {
            'jones': jones_poly,
            'homfly': homfly,
            'stability': float(self.mersenne) / (2**self.p)
        }

    def stability_check(self):
        """
        Verifica se a trança pode sustentar a densidade.
        Equilíbrio entre a potência de 2 e o número de Mersenne.
        """
        # Ratio 2^(p-1) / (2^p - 1) approx 0.5
        ratio = float(2**(self.p-1)) / self.mersenne
        return ratio > 0.49 and ratio < 0.51
