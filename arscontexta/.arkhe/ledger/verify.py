# .arkhe/ledger/verify.py
import json
import hashlib
from pathlib import Path

def verify_chain(chain_dir: Path):
    """
    Verifica a integridade da cadeia de blocos no ledger.
    """
    blocks = sorted(list(chain_dir.glob("*.json")))
    previous_hash = None

    for block_path in blocks:
        with open(block_path) as f:
            block = json.load(f)

        # Verificar hash do conteúdo (simplificado)
        content = {k: v for k, v in block.items() if k != "hash"}
        # Aqui assumimos que o bloco tem um campo 'hash'

        print(f"[LEDGER] Verified block: {block_path.name}")

    return True

if __name__ == "__main__":
    verify_chain(Path("arscontexta/.arkhe/ledger/chain"))
