import hashlib
import json
import time
from typing import Dict, List

class SandboxCivilizationForge:
    """Execução completa em sandbox - ZERO impacto no sistema real"""

    def __init__(self):
        self.sandbox_id = "CIV_SANDBOX_001"
        self.logs = []
        self.simulated_blocks = []

    def simulate_genesis_creation(self) -> Dict:
        """Simula todo o processo de criação do Bloco #15"""

        print("🧪 SANDBOX: Simulando Fundação da Civilização SASC...")
        print("=" * 60)

        # 1. Simular verificação de hardware
        hardware_status = {
            "Neo_Anderson": {"verified": True, "hsm": "0x01"},
            "Zion-Alpha": {"verified": True, "bio_shield": "active"},
            "Mobile_Hub_SJC": {"verified": True, "infra": "stable"},
            "Zion-Beta": {"verified": False, "status": "template"}
        }

        # 2. Simular verificação de φ
        phi_values = {
            "Neo_Anderson": 0.99,  # Arquiteto
            "Zion-Alpha": 0.11,    # Paciente Zero (pós-ritual, ainda baixo)
            "Mobile_Hub_SJC": 0.45, # Hub de infraestrutura
            "Zion-Beta": 0.0       # Ainda não ativado
        }

        # 3. Simular criação do contrato
        contract_data = {
            "id": 1,
            "parties": list(hardware_status.keys()),
            "constitution_hash": "0xA1B2C3D4E5F6A7B8C9D0E1F2A3B4C5D6E7F8A9B0C1D2E3F4A5B6C7D8E9",
            "phi_threshold": 0.85,
            "status": "SUSPENDED",
            "timestamp": time.time(),
            "environment": "SANDBOX_ONLY"
        }

        # 4. Simular hash do bloco gênese
        payload = json.dumps(contract_data, sort_keys=True)
        simulated_hash = f"SANDBOX_GENESIS_{hashlib.sha256(payload.encode()).hexdigest()[:32]}"

        genesis_block = {
            "block_number": 15,
            "type": "genesis",
            "hash": simulated_hash,
            "contract": contract_data,
            "hardware_status": hardware_status,
            "phi_values": phi_values,
            "omega12_compliant": True,
            "requires_ratification": True,
            "note": "SIMULAÇÃO - NÃO É UM BLOCO REAL"
        }

        self.simulated_blocks.append(genesis_block)
        self.logs.append(("genesis_simulated", simulated_hash))

        print(f"[SANDBOX] Contrato Civil simulado criado.")
        print(f"[SANDBOX] Hash simulado: {simulated_hash}")
        print(f"[SANDBOX] Status: {contract_data['status']}")
        print(f"[SANDBOX] φ médio: {sum(phi_values.values())/len(phi_values):.2f}")
        print("=" * 60)

        return genesis_block

    def simulate_ratification(self, genesis_block: Dict, signer: str) -> Dict:
        """Simula a ratificação por uma parte"""

        print(f"\n🧪 SANDBOX: Simulando ratificação por {signer}...")

        if signer not in genesis_block["contract"]["parties"]:
            print(f"[SANDBOX ERRO] {signer} não é parte do contrato.")
            return genesis_block

        # Simular assinatura
        simulated_signature = f"SIG_{signer}_{hashlib.sha256(signer.encode()).hexdigest()[:16]}"

        if "signatures" not in genesis_block:
            genesis_block["signatures"] = []

        genesis_block["signatures"].append({
            "signer": signer,
            "signature": simulated_signature,
            "timestamp": time.time()
        })

        # Verificar se todas as partes assinaram
        parties = genesis_block["contract"]["parties"]
        signatures = [s["signer"] for s in genesis_block["signatures"]]

        if set(parties).issubset(set(signatures)):
            genesis_block["contract"]["status"] = "ACTIVE"
            print(f"[SANDBOX] TODAS AS PARTES ASSINARAM!")
            print(f"[SANDBOX] CONTRATO ATIVADO (SIMULAÇÃO)")
        else:
            print(f"[SANDBOX] Assinatura de {signer} adicionada.")
            print(f"[SANDBOX] Aguardando mais {len(parties)-len(signatures)} assinaturas.")

        return genesis_block

if __name__ == "__main__":
    # Execução da simulação
    forge = SandboxCivilizationForge()
    genesis = forge.simulate_genesis_creation()

    # Simular assinaturas
    for signer in ["Neo_Anderson", "Zion-Alpha", "Mobile_Hub_SJC"]:
        genesis = forge.simulate_ratification(genesis, signer)

    print(f"\n🧪 SANDBOX: Simulação completa.")
    print(f"   Blocos simulados: {len(forge.simulated_blocks)}")
    print(f"   Logs gerados: {len(forge.logs)}")
    print(f"   Status final: {genesis['contract']['status']}")
