import numpy as np
import itertools

class ArkheTrigger:
    """
    Trigger LHC baseado em estrutura de informação temporal Arkhe(n).
    """
    def __init__(self, time_resolution=1e-12):
        self.time_resolution = time_resolution

    def evaluate_event(self, jets_data: list) -> dict:
        """
        jets_data: list of {'pt': float, 'time': float, 'eta': float, 'phi': float}
        """
        n_jets = len(jets_data)
        H_matrix = np.zeros((n_jets, n_jets))

        for i, j in itertools.permutations(range(n_jets), 2):
            dt = jets_data[i]['time'] - jets_data[j]['time']

            # Se dt < 0 (jato i "antes" de j, mas com correlação espacial)
            if dt < -self.time_resolution:
                dR = self.delta_R(jets_data[i], jets_data[j])
                if dR < 0.4:
                    # Score Arkhe(n): correlação não-local temporal
                    H_matrix[i,j] = -np.log10(abs(dt) + 1e-20) * jets_data[i]['pt'] * jets_data[j]['pt'] / (dR**2 + 1e-5)

        arkhe_score = np.sum(H_matrix[H_matrix > 0])
        H_T = sum(j['pt'] for j in jets_data)

        return {
            'arkhe_score': float(arkhe_score / (H_T**2 + 1e-10)),
            'n_violations': int(np.sum(H_matrix > 0)),
            'temporal_structure': H_matrix.tolist()
        }

    def delta_R(self, j1, j2):
        deta = j1['eta'] - j2['eta']
        dphi = j1['phi'] - j2['phi']
        return np.sqrt(deta**2 + dphi**2)
