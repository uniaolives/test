# src/papercoder_kernel/glp/integration.py
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class LinearAToPaperCoder:
    """
    Interface para conectar GLP Linear A ao framework PaperCoder (difeomorfismos).
    """
    def __init__(self, glp_model):
        self.glp = glp_model

    def extract_manifold(self, sign_ids):
        """
        Extrai variedade de representação para análise de difeomorfismos.
        """
        self.glp.eval()
        with torch.no_grad():
            outputs = self.glp(sign_ids, return_wavefunction=True)

        return {
            'tablet_repr': outputs['tablet_repr'].cpu().numpy(),
            'integrated_state': outputs['integrated_state'].cpu().numpy(),
            'scale_probabilities': outputs['scale_probabilities'].cpu().numpy(),
            'well_states': [ws.cpu().numpy() for ws in outputs['well_states']]
        }

    def procrustes_alignment(self, A, B):
        """
        Alinhamento Procrustes para encontrar transformação entre duas estruturas (tabuletas).
        """
        # A, B: [dim] or [seq, dim]
        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch for alignment: {A.shape} vs {B.shape}")

        # Centering
        A_centered = A - np.mean(A, axis=0)
        B_centered = B - np.mean(B, axis=0)

        # SVD for rotation
        # A_centered @ R approx B_centered
        U, S, Vt = np.linalg.svd(A_centered.T @ B_centered)
        R = U @ Vt

        return R # Matriz de rotação (difeomorfismo linear no espaço de embedding)

    def compute_grammar_diffeomorphism(self, sign_ids_a, sign_ids_b):
        """
        Computa a transformação (difeomorfismo) que mapeia a estrutura de A para B.
        """
        man_a = self.extract_manifold(sign_ids_a)
        man_b = self.extract_manifold(sign_ids_b)

        # Usamos a representação global (tablet_repr) ou local (integrated_state)
        # Para o protótipo, usamos a representação global
        R = self.procrustes_alignment(man_a['tablet_repr'], man_b['tablet_repr'])

        return R
