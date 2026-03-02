# examples/arkhe_pillars_demo.py
import sys
import os
from datetime import datetime

# Add core/python to sys.path to import arkhe
sys.path.append(os.path.join(os.getcwd(), 'core', 'python'))

from arkhe.desci.protocol import DeSciManifold, Experiment
from arkhe.consensus.quantum_ising import Validador, IsingConsensusEngine
from arkhe.crypto.pq_arkhe import ArkheSecureNetwork

def run_demo():
    print("--- 🜏 ARKHE PILLARS DEMO ---")

    # 1. CONSENSO POR CRITICIDADE
    print("\n[1] Consenso por Criticidade (Ising Quântico)")
    validadores = [
        Validador("Node-A", 1000.0, 0.99),
        Validador("Node-B", 800.0, 0.95),
        Validador("Node-C", 1200.0, 0.98),
        Validador("Node-D", 500.0, 0.90)
    ]
    engine = IsingConsensusEngine(validadores)

    proposta = b"BLOCK_9245_PROPOSAL"
    success, metadata = engine.alcancar_consensus(proposta)

    print(f"Resultado do Consenso: {metadata['decisao']}")
    print(f"Iterações: {metadata['iteracoes']}")
    m_final = metadata.get('magnetizacao') or metadata.get('magnetizacao_final', 0.0)
    print(f"Magnetização Final: {m_final:.4f}")

    # 2. CRIPTOGRAFIA PÓS-QUÂNTICA
    print("\n[2] Criptografia Pós-Quântica (Kyber + Anomaly Detection)")
    network = ArkheSecureNetwork("Local-Node")
    peer_id = "Remote-Peer"

    if network.establish_channel(peer_id):
        print(f"Canal seguro estabelecido com {peer_id}")

        msg_text = b"DADOS_SENSIVEIS_ARKHE"
        channel = network.channels[peer_id]

        # Simular envio e recebimento
        secure_packet = channel.send_message(msg_text)
        decrypted_msg, anomaly_report = channel.receive_message(secure_packet)

        print(f"Mensagem decifrada: {decrypted_msg.decode()}")
        print(f"Status do Monitor de Anomalias: {anomaly_report['status']}")

    # 3. PROTOCOLO DeSci
    print("\n[3] Protocolo DeSci (Φ-Score)")
    manifold = DeSciManifold()

    # Pesquisador pré-registra experimento
    exp = Experiment(
        experiment_id="",
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

    # Registrar um segundo experimento para permitir conectividade semântica
    exp2 = Experiment(
        experiment_id="",
        title="Estudo de controle sobre percepção estética",
        authors=["Dr. C. Gamma"],
        institution="Instituto de Pesquisa Crítica",
        hypothesis="A percepção de beleza é subjetiva e não segue padrões geométricos",
        methodology_hash="sha256:xyz789...",
        primary_endpoint="score_subjetivo",
        statistical_plan={
            "planned_n": 500,
            "covariates": ["idade"],
            "planned_analysis": "regression",
            "power": 0.80
        },
        pre_registered_at=datetime.now()
    )
    exp2_id = manifold.pre_register_experiment(exp2)
    manifold.submit_results(
        exp2_id,
        data_hash="sha256:ghi012...",
        results={"primary_endpoint_result": 0.5, "actual_sample_size": 500, "analysis_method": "regression", "tested_hypothesis": exp2.hypothesis, "degrees_of_freedom": 499},
        execution_logs=[{"op": "run"}]
    )

    # Submeter resultados
    results = {
        "primary_endpoint_result": 0.72,
        "effect_size_cohen_d": 0.45,
        "p_value": 0.003,
        "actual_sample_size": 987,
        "analysis_method": "mixed_effects_model",
        "tested_hypothesis": exp.hypothesis,
        "degrees_of_freedom": 986
    }

    compliance = manifold.submit_results(
        exp_id,
        data_hash="sha256:def456...",
        results=results,
        execution_logs=[{"op": "randomization"}, {"op": "data_collection"}, {"op": "analysis"}]
    )

    print(f"Score de conformidade: {compliance:.2f}")

    # Calcular Φ-score
    phi_data = manifold.calculate_phi_score(exp_id)
    print(f"Φ-score: {phi_data['phi_score']:.3f}")
    print(f"Interpretação: {phi_data['interpretation']}")

    print("\n--- DEMO FINALIZADA COM SUCESSO ---")

if __name__ == "__main__":
    run_demo()
