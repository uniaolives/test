# src/papercoder_kernel/merkabah/kernel.py
import numpy as np
from typing import Dict, List, Any, Callable, Union

class KernelBridge:
    """
    Camada Κ (Kappa): conecta as camadas do MERKABAH-7 via teoria de kernels.
    Cada camada define uma medida de similaridade (kernel) que induz um RKHS.
    """

    def __init__(self):
        self.kernels = {
            'A_hardware': self._latency_kernel,
            'B_simulation': self._glp_kernel,
            'Φ_crystalline': self._coherence_kernel,
            'Γ_pineal': self._transduction_kernel
        }

    def _latency_kernel(self, x: Any, y: Any) -> float:
        """Kernel exponencial baseado em latência (ms)."""
        # x, y are expected to have a 'latency' attribute or key
        lat_x = x.get('latency', 0.0) if isinstance(x, dict) else getattr(x, 'latency', 0.0)
        lat_y = y.get('latency', 0.0) if isinstance(y, dict) else getattr(y, 'latency', 0.0)
        diff = abs(lat_x - lat_y)
        return float(np.exp(-diff / 10.0))

    def _glp_kernel(self, x: Any, y: Any) -> float:
        """Kernel RBF sobre representações de função de onda GLP."""
        # x, y are expected to have 'wavefunction' and 'coherence'
        wf_x = x.get('wavefunction') if isinstance(x, dict) else getattr(x, 'wavefunction', None)
        wf_y = y.get('wavefunction') if isinstance(y, dict) else getattr(y, 'wavefunction', None)
        coh_x = x.get('coherence', 1.0) if isinstance(x, dict) else getattr(x, 'coherence', 1.0)

        if wf_x is None or wf_y is None:
            return 0.0

        # Convert torch to numpy if needed
        import torch
        if torch.is_tensor(wf_x): wf_x = wf_x.detach().cpu().numpy()
        if torch.is_tensor(wf_y): wf_y = wf_y.detach().cpu().numpy()

        diff = np.linalg.norm(wf_x.flatten() - wf_y.flatten())
        return float(np.exp(-diff**2 / (2 * coh_x**2 + 1e-8)))

    def _coherence_kernel(self, x: Any, y: Any) -> float:
        """Kernel de coerência quântica (fidelidade)."""
        wf_x = x.get('wavefunction') if isinstance(x, dict) else getattr(x, 'wavefunction', None)
        wf_y = y.get('wavefunction') if isinstance(y, dict) else getattr(y, 'wavefunction', None)

        if wf_x is None or wf_y is None:
            return 0.0

        import torch
        if torch.is_tensor(wf_x): wf_x = wf_x.detach().cpu().numpy()
        if torch.is_tensor(wf_y): wf_y = wf_y.detach().cpu().numpy()

        # Fidelity |<psi|phi>|^2
        overlap = np.abs(np.vdot(wf_x.flatten(), wf_y.flatten()))
        return float(overlap**2)

    def _transduction_kernel(self, x: Any, y: Any) -> float:
        """Kernel de resposta a estímulos (Gamma)."""
        sig_x = x.get('signal', 0.0) if isinstance(x, dict) else getattr(x, 'signal', 0.0)
        sig_y = y.get('signal', 0.0) if isinstance(y, dict) else getattr(y, 'signal', 0.0)
        return float(np.exp(-abs(sig_x - sig_y)))

    def combine_kernels(self, weights: Dict[str, float]) -> Callable[[Any, Any], float]:
        """
        Combinação convexa de kernels (multi-view learning).
        """
        def combined_kernel(x: Dict[str, Any], y: Dict[str, Any]) -> float:
            value = 0.0
            for name, kernel_fn in self.kernels.items():
                if name in x and name in y:
                    value += weights.get(name, 0.0) * kernel_fn(x[name], y[name])
            return value
        return combined_kernel

    def _compute_gram_matrix(self, states: List[Any], kernel_name: str) -> np.ndarray:
        """Computa a matriz de Gram para um conjunto de estados."""
        N = len(states)
        K = np.zeros((N, N))
        kernel_fn = self.kernels.get(kernel_name)
        if not kernel_fn:
            raise ValueError(f"Kernel {kernel_name} not found")

        for i in range(N):
            for j in range(i, N):
                val = kernel_fn(states[i], states[j])
                K[i, j] = K[j, i] = val
        return K

    def kernel_pca(self, states: List[Any], kernel_name: str = 'Φ_crystalline'):
        """
        Aplica Kernel PCA para extrair componentes principais no espaço de Hilbert.
        """
        K = self._compute_gram_matrix(states, kernel_name)

        # Centralizar a matriz de Gram
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        # Autovalores / autovetores
        eigvals, eigvecs = np.linalg.eigh(K_centered)

        # Ordenar decrescentemente
        idx = np.argsort(eigvals)[::-1]
        return eigvals[idx], eigvecs[:, idx]
