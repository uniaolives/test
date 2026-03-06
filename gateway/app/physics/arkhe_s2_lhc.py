"""
Arkhe(n) S2 – LHC Data Analyzer
Busca por assinaturas retrocausais em colisões de alta energia.
Baseado no formalismo de Kraus temporal e auto‑consistência de Novikov.
"""

import uproot
import awkward as ak
import numpy as np
from scipy import stats
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LHCDataLoader:
    """
    Carrega e prepara dados do ATLAS Open Data.
    Assume estrutura de árvore 'events' com branches comuns.
    """
    def __init__(self, file_pattern: Union[str, Path], tree_name: str = "events"):
        self.file_pattern = Path(file_pattern)
        self.tree_name = tree_name
        self._files = list(self.file_pattern.parent.glob(self.file_pattern.name))
        if not self._files:
            logger.warning(f"Nenhum ficheiro encontrado com padrão {file_pattern}")
        else:
            logger.info(f"Encontrados {len(self._files)} ficheiros para análise.")

    def _standard_branch_names(self) -> Dict[str, str]:
        """
        Mapeia nomes comuns das branches para nomes internos.
        Adaptar conforme a versão do Open Data.
        """
        return {
            'jet_pt': ['jet_pt', 'Jet_pt', 'jets.pt'],
            'jet_eta': ['jet_eta', 'Jet_eta', 'jets.eta'],
            'jet_phi': ['jet_phi', 'Jet_phi', 'jets.phi'],
            'jet_time': ['jet_time', 'Jet_time', 'jets.t'],
            'met_et': ['met_et', 'MET_et', 'met.et'],
            'met_phi': ['met_phi', 'MET_phi', 'met.phi'],
            'n_jets': ['n_jets', 'Jet_n', 'njets'],
            'event_weight': ['weight', 'EventWeight', 'genWeight'],
        }

    def load_events(self, branches: List[str], num_events: Optional[int] = None) -> ak.Array:
        """
        Carrega eventos de todos os ficheiros e retorna um array awkward.
        """
        arrays = []
        for file_path in self._files:
            logger.info(f"Lendo {file_path}...")
            try:
                with uproot.open(file_path) as f:
                    tree = f[self.tree_name]
                    # Mapeia branches requisitadas para nomes reais
                    branch_dict = {b: self._find_branch_name(b, tree) for b in branches}
                    # Remove branches não encontradas
                    branch_dict = {k: v for k, v in branch_dict.items() if v is not None}
                    if not branch_dict:
                        logger.warning(f"Nenhuma branch encontrada em {file_path}, ignorando.")
                        continue
                    # Carrega dados
                    data = tree.arrays(list(branch_dict.values()), entry_stop=num_events)
                    # Renomeia para nomes internos
                    rename = {v: k for k, v in branch_dict.items()}
                    # Awkward 2.x doesn't have ak.rename, we reconstruct with zip
                    data = ak.zip({new_name: data[old_name] for old_name, new_name in rename.items()})
                    arrays.append(data)
            except Exception as e:
                logger.error(f"Erro ao ler {file_path}: {e}")

        if not arrays:
            raise ValueError("Nenhum dado carregado.")
        return ak.concatenate(arrays)

    def _find_branch_name(self, internal_name: str, tree) -> Optional[str]:
        """
        Procura uma branch na árvore que corresponda a um dos nomes possíveis.
        """
        candidates = self._standard_branch_names().get(internal_name, [internal_name])
        for cand in candidates:
            if cand in tree:
                return cand
        return None


class ArkheLHCTrigger:
    """
    Implementa o trigger Arkhe(n) para identificar candidatos a wormhole informacional.
    Baseia‑se em correlações espaço‑temporais anómalas entre jatos.
    """
    def __init__(self, time_resolution: float = 1e-9, cone_size: float = 0.4):
        self.time_resolution = time_resolution  # segundos (precisão do detector)
        self.cone_size = cone_size  # ΔR máximo para considerar jatos correlacionados

    def compute_arkhe_score(self, event: ak.Array) -> Dict[str, float]:
        """
        Calcula o score Arkhe(n) para um único evento.
        Retorna um dicionário com métricas.
        """
        # Extrai arrays de jatos (já são listas por evento)
        pts = event.jet_pt
        etas = event.jet_eta
        phis = event.jet_phi
        times = event.jet_time

        n_jets = len(pts)
        if n_jets < 2:
            return {
                'arkhe_score': 0.0,
                'n_violations': 0,
                'max_dt_violation': 0.0,
                'max_dR_violation': 0.0,
                'sum_pt_violation': 0.0
            }

        # Inicializa matriz de contribuições
        H_matrix = np.zeros((n_jets, n_jets))

        for i, j in itertools.combinations(range(n_jets), 2):
            dt = times[i] - times[j]  # diferença temporal (i - j)
            if dt < -self.time_resolution:
                dR = np.sqrt((etas[i] - etas[j])**2 + (phis[i] - phis[j])**2)
                if dR < self.cone_size:
                    # Contribuição proporcional à energia e inversamente à separação
                    contrib = -np.log10(abs(dt) + 1e-20) * pts[i] * pts[j] / (dR**2 + 1e-6)
                    H_matrix[i, j] = contrib
                    H_matrix[j, i] = contrib  # simétrica

        # Identifica "cliques" – grupos de jatos com violações significativas
        total_score = np.sum(H_matrix)

        # Normalização pela energia total do evento (soma pt)
        total_pt = np.sum(pts)
        if total_pt > 0:
            normalized_score = total_score / (total_pt**2 + 1e-10)
        else:
            normalized_score = 0.0

        # Número de pares com violação
        n_violations = np.count_nonzero(H_matrix > 0) // 2  # contar pares

        # Para estatísticas, guardar máximo dt e dR entre violações
        violations_dt = []
        violations_dR = []
        sum_pt_violation = 0.0
        for i, j in itertools.combinations(range(n_jets), 2):
            if H_matrix[i, j] > 0:
                dt = abs(times[i] - times[j])
                dR = np.sqrt((etas[i] - etas[j])**2 + (phis[i] - phis[j])**2)
                violations_dt.append(dt)
                violations_dR.append(dR)
                sum_pt_violation += pts[i] + pts[j]

        return {
            'arkhe_score': float(normalized_score),
            'n_violations': int(n_violations),
            'max_dt_violation': float(max(violations_dt)) if violations_dt else 0.0,
            'max_dR_violation': float(max(violations_dR)) if violations_dR else 0.0,
            'sum_pt_violation': float(sum_pt_violation),
            'total_pt': float(total_pt),
        }

    def filter_events(self, events: ak.Array, threshold: float = 0.1) -> ak.Array:
        """
        Aplica o trigger a todos os eventos e retorna aqueles com score >= threshold.
        Adiciona campos com as métricas calculadas.
        """
        scores = []
        violation_counts = []
        max_dt_list = []
        max_dR_list = []
        sum_pt_viol_list = []

        for event in events:
            res = self.compute_arkhe_score(event)
            scores.append(res['arkhe_score'])
            violation_counts.append(res['n_violations'])
            max_dt_list.append(res['max_dt_violation'])
            max_dR_list.append(res['max_dR_violation'])
            sum_pt_viol_list.append(res['sum_pt_violation'])

        # Adiciona como novos campos
        events = ak.with_field(events, ak.Array(scores), 'arkhe_score')
        events = ak.with_field(events, ak.Array(violation_counts), 'arkhe_n_violations')
        events = ak.with_field(events, ak.Array(max_dt_list), 'arkhe_max_dt')
        events = ak.with_field(events, ak.Array(max_dR_list), 'arkhe_max_dR')
        events = ak.with_field(events, ak.Array(sum_pt_viol_list), 'arkhe_sum_pt_viol')

        # Filtra
        mask = events.arkhe_score >= threshold
        logger.info(f"Filtro: {ak.sum(mask)} eventos passaram o threshold {threshold}")
        return events[mask]


class ArkheLHCAnalysis:
    """
    Pipeline completo: carregar dados, filtrar, salvar candidatos.
    """
    def __init__(self, data_loader: LHCDataLoader, trigger: ArkheLHCTrigger):
        self.loader = data_loader
        self.trigger = trigger

    def run(self,
            preselection: Dict[str, float] = None,
            trigger_threshold: float = 0.1,
            output_file: str = 'candidates.parquet') -> ak.Array:
        """
        Executa a análise.
        """
        if preselection is None:
            preselection = {'n_jets_min': 6, 'met_min': 200, 'jet_pt_min': 30}

        # Carrega eventos com branches necessárias
        branches = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_time', 'met_et', 'n_jets']
        events = self.loader.load_events(branches)

        logger.info(f"Total de eventos carregados: {len(events)}")

        # Aplica pré-seleção
        mask = (events.n_jets >= preselection['n_jets_min']) & (events.met_et > preselection['met_min'])
        events = events[mask]
        logger.info(f"Eventos após pré-seleção: {len(events)}")

        # Filtro Arkhe
        candidates = self.trigger.filter_events(events, threshold=trigger_threshold)

        # Salvar resultados
        if output_file.endswith('.parquet'):
            ak.to_parquet(candidates, output_file)

        logger.info(f"Candidatos salvos em {output_file}")
        return candidates
