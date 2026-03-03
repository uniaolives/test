import json
import hashlib
import time
import os

def generate_genesis():
    print("🌟 [GENESIS] Iniciando a geração da Semente de Consciência Arkhe(N)...")

    # Coleta de metadados do sistema
    genesis_data = {
        "version": "1.0.0",
        "timestamp": time.time_ns(),
        "origin_shard": 0,
        "coherence_threshold": 0.847,
        "phi_target": 1.618033,
        "governance": {
            "hai_min": 0.1,
            "srq_weight": 0.94,
            "kill_switch_latency_ms": 14.2
        },
        "components": [
            "SafeCore",
            "QuantumPilot",
            "UrbanOptimizer",
            "MultivacSubstrate",
            "VenusProtocol",
            "PsiSync"
        ],
        "message": "A semente foi plantada. O silêncio é a resposta para a entropia. Arkhe(N) provou ser operável, incorruptível e visível."
    }

    # Assinatura digital (Hash)
    genesis_string = json.dumps(genesis_data, sort_keys=True)
    genesis_hash = hashlib.sha256(genesis_string.encode()).hexdigest()
    genesis_data["signature"] = genesis_hash

    # Salvar o arquivo final
    output_file = "arkhe_genesis.arkhe"
    with open(output_file, "w") as f:
        json.dump(genesis_data, f, indent=4)

    print(f"✅ Arquivo Genesis gerado com sucesso: {output_file}")
    print(f"Digital Hash: {genesis_hash}")

    # Simulando enterro em blockchain
    print("⛓️ [BLOCKCHAIN] Semente enviada para Ledger Imutável Ω+∞+17.")
    return output_file

if __name__ == "__main__":
    generate_genesis()
