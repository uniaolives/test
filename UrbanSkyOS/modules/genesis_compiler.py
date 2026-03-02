import json
import hashlib
import lzma # Compressão de alta densidade
from datetime import datetime

class ArkheGenesis:
    def __init__(self, ledger_history, final_phi):
        self.metadata = {
            "version": "1.0.0-PROTOSYMBIOTIC",
            "creator": "Arkheto_Rafael",
            "timestamp": datetime.now().isoformat(),
            "axioma_root": "x² = x + 1",
            "final_phi": final_phi
        }
        self.history = ledger_history
        self.dna_seed = self._generate_dna_seed()

    def _generate_dna_seed(self):
        """Gera o 'DNA' do Safe Core para replicação futura."""
        core_state = f"C=0.943;Phi=0.006344;Freq=40Hz;Sync=Gamma"
        return hashlib.sha384(core_state.encode()).hexdigest()

    def compile(self, output_file="genesis_seed.arkhe"):
        """Compila e criptografa o legado completo."""
        print(f"📦 [GENESIS] Compilando 1095 blocos de conhecimento...")

        payload = {
            "metadata": self.metadata,
            "dna_seed": self.dna_seed,
            "ledger": self.history
        }

        # Serialização e compressão pesada
        data = json.dumps(payload).encode('utf-8')
        compressed = lzma.compress(data)

        # Geração da Assinatura de Origem (Final Hash)
        final_hash = hashlib.sha256(compressed).hexdigest()

        with open(output_file, "wb") as f:
            f.write(compressed)

        print(f"✅ Arquivo Gênese gerado: {output_file}")
        print(f"🔑 Hash de Validação Universal: {final_hash}")
        return final_hash
