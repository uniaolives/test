import numpy as np
from scipy.stats import mannwhitneyu


class GeometricValidator:
    """
    Minimal geometric/information validator:
    - enforces accuracy + entropy thresholds
    - compares two representation sets via a simple distance statistic
    """

    def __init__(self, min_acc=0.85, entropy_threshold=4.0):
        self.min_acc = float(min_acc)
        self.entropy_threshold = float(entropy_threshold)

    # --- High-level checks (used from daemon or C-bridge) ---

    def check_convergence(self, acc: float, entropy: float, threshold: float) -> bool:
        if acc < self.min_acc:
            return False
        if entropy > min(threshold, self.entropy_threshold):
            return False
        return True

    def verify_information_conservation(self,
                                        states_a: np.ndarray,
                                        states_b: np.ndarray,
                                        targets: np.ndarray):
        """
        Placeholder: compare the pairwise distance distributions of states_a vs states_b.
        Returns (ok, message).
        """
        if states_a.shape != states_b.shape:
            return False, "shape mismatch"

        # Flatten distances as a cheap proxy
        da = np.linalg.norm(states_a - states_a.mean(axis=0, keepdims=True), axis=1)
        db = np.linalg.norm(states_b - states_b.mean(axis=0, keepdims=True), axis=1)

        u, p = mannwhitneyu(da, db, alternative="two-sided")
        # Accept if distributions are not significantly different
        ok = p > 0.05
        msg = f"MWU p={p:.4f}, ok={ok}"
        return ok, msg

    def stratified_basin_sampling(self,
                                  states: np.ndarray,
                                  n_basins: int,
                                  n_per_basin: int) -> np.ndarray:
        """
        Very simple k-means-like stratified sampler:
        - Cluster states along first principal component into n_basins bins
        - Uniformly sample n_per_basin from each bin
        """
        n_states, dim = states.shape
        if n_states < n_basins * n_per_basin:
            raise ValueError("not enough states for requested stratified sample")

        # Project onto first PC via SVD
        u, s, vh = np.linalg.svd(states - states.mean(axis=0, keepdims=True),
                                 full_matrices=False)
        proj = u[:, 0]  # 1D coordinate

        # Bin into quantiles
        quantiles = np.linspace(0.0, 1.0, n_basins + 1)
        edges = np.quantile(proj, quantiles)

        samples = []
        for i in range(n_basins):
            mask = (proj >= edges[i]) & (proj <= edges[i + 1] + 1e-9)
            idx = np.nonzero(mask)[0]
            if len(idx) < n_per_basin:
                # fallback: sample with replacement
                chosen = np.random.choice(idx, size=n_per_basin, replace=True)
            else:
                chosen = np.random.choice(idx, size=n_per_basin, replace=False)
            samples.append(states[chosen])

        return np.concatenate(samples, axis=0)

    def verify_nematic_phase(self, weight_matrix):
        """
        [I13] Verifica se o manifold está em fase de "Cristal Líquido".
        Bio-Analogia: Simula a ordem orientacional dos fagos Pf.
        """
        # 1. Calcula os vetores principais (SVD) de cada camada
        # Estes são nossos "varetas" (phages)
        principal_components = []
        for layer in weight_matrix:
            # A camada deve ser 2D para ter um SVD significativo
            if layer.ndim != 2 or min(layer.shape) == 0:
                continue
            _, _, Vt = np.linalg.svd(layer, full_matrices=False)
            principal_components.append(Vt[0])  # Pega o vetor de maior variação

        if not principal_components:
            return False, "Estado Inválido (sem camadas 2D para analisar)"

        # 2. Calcula o Parâmetro de Ordem Nemático (S)
        S = _compute_nematic_order_parameter(principal_components)

        # 3. A Regra de Ouro (Bio-Análoga)
        if S < 0.3:
            return False, f"Estado Gasoso (S={S:.4f}, Entropia Alta / Não Convergiu)"
        elif S > 0.95:
            return False, f"Estado Cristalino (S={S:.4f}, Overfitting / Quebra frágil)"
        else:
            return True, f"Estado Líquido Nemático (S={S:.4f}, Resiliente / Adaptável)"


def _compute_nematic_order_parameter(vectors):
    """
    Computes the nematic order parameter S for a list of orientation vectors.
    S = d/(d-1) * (lambda_max - 1/d), where lambda_max is the largest eigenvalue
    of the ordering tensor Q = <v . v^T>.
    """
    if not vectors:
        return 0.0

    # Garante que todos os vetores sejam normalizados
    vectors = [v / np.linalg.norm(v) for v in vectors if np.linalg.norm(v) > 0]

    if not vectors:
        return 0.0

    # Obtém a dimensão d
    d = vectors[0].shape[0]
    if d <= 1:
        # O parâmetro de ordem não é bem definido para d=1
        return 1.0

    # Constrói o tensor de ordenação Q
    q_tensor = np.mean([np.outer(v, v) for v in vectors], axis=0)

    # Encontra os autovalores
    eigenvalues = np.linalg.eigvalsh(q_tensor)
    lambda_max = np.max(eigenvalues)

    # Calcula S
    s = (d / (d - 1)) * (lambda_max - (1 / d))
    return s
