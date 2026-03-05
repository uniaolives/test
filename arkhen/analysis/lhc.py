# arkhen/analysis/lhc.py
import uproot
import awkward as ak
import numpy as np
from scipy import stats

class ArkheLHCAnalyzer:
    def __init__(self, data_file=None):
        self.data_file = data_file
        self.branches = [
            "jet_pt", "jet_eta", "jet_phi", "jet_time",
            "met_et", "met_phi",
            "n_jets", "n_vertices"
        ]

    def select_high_multiplicity(self, arrays, n_jets_min=6, met_min=200):
        """
        Seleciona eventos com alta multiplicidade e MET
        (assinaturas de "wormhole informacional")
        """
        mask = (arrays["n_jets"] >= n_jets_min) & \
               (arrays["met_et"] > met_min)

        return {k: v[mask] for k, v in arrays.to_dict().items() if k in self.branches}

    def temporal_correlation_analysis(self, events):
        """
        Busca correlações não-locais temporais entre jatos
        """
        n_events = len(events["jet_pt"])
        correlations = []

        for i in range(n_events):
            times = ak.to_numpy(events["jet_time"][i])
            etas = ak.to_numpy(events["jet_eta"][i])
            phis = ak.to_numpy(events["jet_phi"][i])

            # Matriz de correlação temporal
            dt_matrix = times[:, None] - times[None, :]
            dR_matrix = np.sqrt(
                (etas[:, None] - etas[None, :])**2 +
                (phis[:, None] - phis[None, :])**2
            )

            # Busca pares com dt < 0 (causalidade aparente invertida)
            # mas correlação espacial anômala
            anomalous = (dt_matrix < -1e-9) & (dR_matrix < 0.4)
            if np.any(anomalous):
                correlations.append({
                    'event_id': i,
                    'n_anomalous': np.sum(anomalous),
                    'dt_min': np.min(dt_matrix[anomalous])
                })

        return correlations

    def kraus_inspired_trigger(self, events, threshold=3.0):
        """
        Trigger baseado em estrutura de Kraus temporal:
        busca "squeezing" efetivo nas distribuições
        """
        results = []

        for i in range(len(events["jet_pt"])):
            pts = ak.to_numpy(events["jet_pt"][i])

            # "Squeezing" = variância reduzida em certas direções
            # Medido via momentos de distribuição
            skew_pt = stats.skew(pts)
            kurt_pt = stats.kurtosis(pts)

            # Score Arkhe(n): desvio da distribuição gaussiana
            arkhe_score = abs(skew_pt) + abs(kurt_pt)

            if arkhe_score > threshold:
                results.append({
                    'event_id': i,
                    'arkhe_score': arkhe_score,
                    'n_jets': len(pts)
                })

        return results
