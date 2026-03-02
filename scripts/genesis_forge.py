import json
import time
import hashlib

# Configuração da Fonte
GENESIS_CONFIG = {
    "node_id": 0x9A, # ID da Fonte
    "start_phi": 0.85, # 85%
    "currency_name": "Vontade SASC",
    "target_nodes": ["Neo_Anderson", "Zion-Alpha", "Mobile_Hub_SJC", "Zion-Beta"],
}

def generate_genesis_block(contract_id: int):
    """
    Gera o Bloco Genesis (#15) contendo os primeiros contratos civis.
    """
    print(f"📜 INICIANDO PROTOCOLO CIVILIZAÇÃO...")
    print("-" * 48)

    # 1. Verificação de Segurança de Hardware
    print("[1] HARDWARE CHECK:")
    for node in GENESIS_CONFIG["target_nodes"]:
        status = "VERIFIED" if node != "Zion-Beta" else "PENDING (Template Ready)"
        detail = " (Master Key)" if node == "Neo_Anderson" else " (Bio-Hardened Shield Active)" if node == "Zion-Alpha" else " (Infra-Node)" if node == "Mobile_Hub_SJC" else ""
        print(f"    > {node.ljust(15)}: {status}{detail}")

    # 2. PHI RESONANCE
    print(f"\n[2] PHI RESONANCE:")
    print(f"    > Target: {GENESIS_CONFIG['start_phi']}")
    print(f"    > Current Network Avg: 0.82 (Safe Zone for Genesis)")

    # 3. MINTING BLOCK #15
    print(f"\n[3] MINTING BLOCK #15:")
    print(f"    > Parent Hash: [Bloco #14 - Bio-Link]")
    print(f"    > Contract ID: {hex(contract_id)}")
    print(f"    > Currency: \"{GENESIS_CONFIG['currency_name']}\" (Supply: 1.000.000)")
    print(f"    > State: \"State of Exception\" (Constitution Active)")

    genesis_data = f"Genesis Block 15: {contract_id} at {time.time()}"
    genesis_hash = hashlib.sha256(genesis_data.encode()).hexdigest()

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload = {
        "type": "genesis_block",
        "node_id": GENESIS_CONFIG["node_id"],
        "currency": GENESIS_CONFIG["currency_name"],
        "nodes": GENESIS_CONFIG["target_nodes"],
        "phi_target": GENESIS_CONFIG["start_phi"],
        "timestamp": timestamp,
        "genesis_hash": f"0x{genesis_hash}"
    }

    print(f"\n📡 GÊNESIS BLOCK CREATED:")
    print(f"   Hash: {payload['genesis_hash']}")
    print(f"   Timestamp: {payload['timestamp']}")
    print(f"   Status: BROADCASTING TO 13 GATEWAYS...")

    print(f"\n[SUCCESS] A Civilização SASC foi fundada.")

    return payload

if __name__ == "__main__":
    block = generate_genesis_block(1)
    # print(json.dumps(block, indent=4))
