# arkhe_web3/demo_desci_workflow.py
from desci_protocol import DeSciManifold, Experiment
from datetime import datetime
import numpy as np

# Inicializar manifold
manifold = DeSciManifold()

# Pesquisador pré-registra experimento
exp = Experiment(
    title="Efeito da proporção áurea na percepção de beleza",
    authors=["Dr. A. Arkhe", "Prof. B. Omega"],
    institution="Instituto de Pesquisa Crítica",
    hypothesis="Estímulos com proporção φ = 0.618 são avaliados como mais agradáveis",
    methodology_hash="sha256:abc123...",
    primary_endpoint="score_medio_beleza",
    statistical_plan={
        "planned_n": 1000,
        "covariates": ["idade", "genero", "cultura"],
        "planned_analysis": "mixed_effects_model",
        "power": 0.95
    },
    pre_registered_at=datetime.now()
)

# Registrar
exp_id = manifold.pre_register_experiment(exp)
print(f"Experimento pré-registrado: {exp_id}")

# ... tempo passa, experimento é executado ...

# Submeter resultados
results = {
    "score_medio_beleza": 0.72,
    "effect_size_cohen_d": 0.45,
    "p_value": 0.003,
    "actual_sample_size": 987,
    "analysis_method": "mixed_effects_model",
    "tested_hypothesis": exp.hypothesis,  # conforme pré-registrado
    "degrees_of_freedom": 985
}

compliance = manifold.submit_results(
    exp_id,
    data_hash="sha256:def456...",
    results=results,
    execution_logs=[{"op": "randomization"}, {"op": "data_collection"}, {"op": "analysis"}]
)

print(f"Score de conformidade: {compliance}")

# Calcular Φ-score
phi_data = manifold.calculate_phi_score(exp_id)
print(f"Φ-score: {phi_data['phi_score']:.3f}")
print(f"Componentes: {phi_data['components']}")

# Demonstração de PQC e Consenso (integração visual)
print("\n--- Arkhe Web3 Synthesis Status ---")
print("Consensus Engine: Quantum Ising Mode ACTIVE")
print("PQC Channel: Kyber-512 + Correlation Monitor ACTIVE")
print("DeSci Ledger: Omega Ledger Integration ACTIVE")
