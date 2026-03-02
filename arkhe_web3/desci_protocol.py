# arkhe_web3/desci_protocol.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime
import hashlib
import json

@dataclass
class Experiment:
    """
    Experimento científico registrado no manifold Arkhe.
    """
    experiment_id: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    institution: str = ""

    # Pré-registro (imutável após timestamp)
    hypothesis: str = ""           # Hipótese testada
    methodology_hash: str = ""     # Hash do protocolo completo
    primary_endpoint: str = ""     # Endpoint primário definido a priori
    statistical_plan: dict = field(default_factory=dict) # Plano de análise (evita p-hacking)
    pre_registered_at: datetime = field(default_factory=datetime.now)

    # Execução (preenchido posteriormente)
    data_hash: Optional[str] = None
    results: Optional[dict] = None
    executed_at: Optional[datetime] = None

    # Metadados Arkhe
    entropy_cost: float = 0.0  # AEU gasto na execução
    phi_score: float = 0.0     # Calculado post-hoc

    def compute_integrity_hash(self) -> str:
        """Hash criptográfico do pré-registro (imutável)."""
        content = {
            'hypothesis': self.hypothesis,
            'methodology': self.methodology_hash,
            'endpoint': self.primary_endpoint,
            'statistical_plan': self.statistical_plan,
            'timestamp': self.pre_registered_at.isoformat()
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

@dataclass
class KnowledgeNode:
    """
    Nó no grafo de conhecimento científico.
    """
    paper_id: str
    experiment: Experiment
    citations: List[str] = field(default_factory=list)
    cited_by: List[str] = field(default_factory=list)

    # Vetor de embedding semântico (da hipótese/metodologia)
    embedding: Optional[np.ndarray] = None

    # Métricas de impacto
    replication_attempts: int = 0
    replication_successes: int = 0
    contradiction_score: float = 0.0  # 0 = consistente, 1 = fortemente contraditado

class DeSciManifold:
    """
    Manifold de conhecimento científico com Φ-score como métrica de qualidade.
    """

    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.omega_ledger: List[dict] = []  # Registro imutável

        # Constantes
        self.PHI = 0.618033988749894

    def pre_register_experiment(self, exp: Experiment) -> str:
        """
        Registra experimento antes da execução (pré-registro obrigatório).
        """
        integrity_hash = exp.compute_integrity_hash()

        # Verificar duplicidade
        if any(e.compute_integrity_hash() == integrity_hash for e in self.experiments.values()):
            # find the existing one
            for eid, e in self.experiments.items():
                if e.compute_integrity_hash() == integrity_hash:
                    return eid

        exp.experiment_id = integrity_hash[:16]
        self.experiments[exp.experiment_id] = exp

        # Registrar no Omega Ledger
        ledger_entry = {
            'type': 'EXPERIMENT_PREREGISTERED',
            'timestamp': datetime.now().isoformat(),
            'experiment_id': exp.experiment_id,
            'integrity_hash': integrity_hash,
            'authors': exp.authors,
            'institution': exp.institution,
            'aeu_cost_estimate': self._estimate_entropy_cost(exp)
        }
        self.omega_ledger.append(ledger_entry)

        return exp.experiment_id

    def _estimate_entropy_cost(self, exp: Experiment) -> float:
        """
        Estima o custo entropico de um experimento baseado em complexidade.
        """
        # Fatores: número de variáveis, tamanho da amostra, complexidade estatística
        complexity_factors = {
            'variables': len(exp.statistical_plan.get('covariates', [])),
            'interactions': exp.statistical_plan.get('max_interaction_order', 1),
            'sample_size': exp.statistical_plan.get('planned_n', 100),
            'endpoints': len(exp.statistical_plan.get('secondary_endpoints', [])) + 1
        }

        # Entropia de Shannon estimada do design experimental
        entropy = np.log2(complexity_factors['sample_size']) * max(1, complexity_factors['variables'])
        entropy *= complexity_factors['interactions'] * complexity_factors['endpoints']

        return entropy / 1e6  # normalizado para AEU

    def submit_results(self, experiment_id: str, data_hash: str,
                      results: dict, execution_logs: List[dict]):
        """
        Submeter resultados de experimento executado.
        """
        if experiment_id not in self.experiments:
            raise ValueError("Experimento não pré-registrado")

        exp = self.experiments[experiment_id]
        exp.data_hash = data_hash
        exp.results = results
        exp.executed_at = datetime.now()
        exp.entropy_cost = self._calculate_actual_entropy(results, execution_logs)

        # Verificar conformidade com pré-registro (anti-p-hacking)
        compliance_score = self._verify_compliance(exp)

        # Criar nó no grafo de conhecimento
        node = KnowledgeNode(
            paper_id=experiment_id,
            experiment=exp,
            embedding=self._compute_embedding(exp.hypothesis)
        )
        self.knowledge_graph[experiment_id] = node

        # Registrar
        self.omega_ledger.append({
            'type': 'RESULTS_SUBMITTED',
            'timestamp': datetime.now().isoformat(),
            'experiment_id': experiment_id,
            'compliance_score': compliance_score,
            'entropy_cost_actual': exp.entropy_cost,
            'data_integrity': data_hash
        })

        return compliance_score

    def _verify_compliance(self, exp: Experiment) -> float:
        """
        Verifica se os resultados seguem o protocolo pré-registrado.
        Score 1.0 = total conformidade, 0.0 = desvio grave (possível p-hacking).
        """
        score = 1.0

        if not exp.results: return 0.0

        # Verificar se endpoint primário foi reportado
        if exp.primary_endpoint not in exp.results:
            score -= 0.3

        # Verificar se análise foi conforme plano estatístico
        if exp.results.get('analysis_method') != exp.statistical_plan.get('planned_analysis'):
            score -= 0.2

        # Verificar se amostra está dentro do planejado
        actual_n = exp.results.get('actual_sample_size', 0)
        planned_n = exp.statistical_plan.get('planned_n', actual_n)
        if actual_n > planned_n * 1.2:  # tolerância de 20%
            score -= 0.3

        # Verificar se hipótese foi alterada post-hoc
        if exp.results.get('tested_hypothesis') != exp.hypothesis:
            score -= 0.5

        return max(0.0, score)

    def _calculate_actual_entropy(self, results: dict, logs: List[dict]) -> float:
        n_operations = len(logs)
        data_complexity = results.get('degrees_of_freedom', 1)
        return n_operations * np.log2(max(1.1, data_complexity)) / 1e6

    def _compute_embedding(self, text: str) -> np.ndarray:
        # Em produção: usar sentence-transformers
        state = np.random.get_state()
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(128)
        np.random.set_state(state)
        return emb

    def calculate_phi_score(self, experiment_id: str) -> dict:
        if experiment_id not in self.knowledge_graph:
            raise ValueError("Experimento não encontrado no grafo")

        node = self.knowledge_graph[experiment_id]
        exp = node.experiment

        Q_intrinsic = max(0.1, exp.entropy_cost) * exp.statistical_plan.get('power', 0.8)
        Q_intrinsic *= self._verify_compliance(exp)

        C_semantic = self._calculate_semantic_connectivity(node)

        R_repro = np.log1p(node.replication_successes) / np.log1p(node.replication_attempts + 1)

        I_citation = self._calculate_citation_impact(node)

        phi_weights = [self.PHI, 1-self.PHI, self.PHI**2, (1-self.PHI)**2]
        phi_weights = np.array(phi_weights) / sum(phi_weights)

        components = np.array([Q_intrinsic, C_semantic, R_repro, I_citation])
        # avoid log of zero
        components = np.clip(components, 1e-6, None)
        phi_score = np.prod(components ** phi_weights)

        phi_score = np.tanh(phi_score * 10) # scale

        node.experiment.phi_score = phi_score

        return {
            'experiment_id': experiment_id,
            'phi_score': float(phi_score),
            'components': {
                'intrinsic_quality': float(Q_intrinsic),
                'semantic_connectivity': float(C_semantic),
                'reproducibility': float(R_repro),
                'citation_impact': float(I_citation)
            }
        }

    def _calculate_semantic_connectivity(self, node: KnowledgeNode) -> float:
        if node.embedding is None or len(self.knowledge_graph) < 2:
            return 0.1

        similarities = []
        for other_id, other_node in self.knowledge_graph.items():
            if other_id == node.paper_id or other_node.embedding is None:
                continue
            sim = np.dot(node.embedding, other_node.embedding) / (
                np.linalg.norm(node.embedding) * np.linalg.norm(other_node.embedding) + 1e-10
            )
            similarities.append(sim)

        if not similarities: return 0.1

        sim_array = np.array(similarities)
        connectivity = np.std(sim_array) * np.mean(sim_array) + 0.1

        return float(connectivity)

    def _calculate_citation_impact(self, node: KnowledgeNode) -> float:
        if not node.cited_by: return 0.1

        total_impact = 0.0
        for citing_id in node.cited_by:
            if citing_id in self.knowledge_graph:
                citing_phi = self.knowledge_graph[citing_id].experiment.phi_score
                total_impact += citing_phi

        return (np.log1p(total_impact) / np.log1p(len(node.cited_by))) + 0.1
