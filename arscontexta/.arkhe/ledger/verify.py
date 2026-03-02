# .arkhe/ledger/verify.py
import json
import hashlib
from pathlib import Path

def compute_hash(data: dict) -> str:
    """Computa o hash SHA-256 de um dicionário (excluindo o campo 'hash')."""
    content = {k: v for k, v in data.items() if k != "hash"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True).encode()
    ).hexdigest()

def verify_chain(chain_dir: Path):
    """
    Verifica a integridade da cadeia de blocos no ledger.
    """
    blocks = sorted(list(chain_dir.glob("*.json")))
    previous_hash = None

    for block_path in blocks:
        with open(block_path) as f:
            block = json.load(f)

        # 1. Verificar hash interno
        computed = compute_hash(block)
        if "hash" in block and block["hash"] != computed:
            print(f"[FAILED] Block {block_path.name}: Internal hash mismatch.")
            return False

        # 2. Verificar encadeamento (exceto genesis)
        if previous_hash and "parent_hash" in block:
            if block["parent_hash"] != previous_hash:
                print(f"[FAILED] Block {block_path.name}: Chain link broken.")
                return False

        # O genesis global não tem parent_hash, mas local_genesis sim.
        # Guardar hash para o próximo bloco
        previous_hash = block.get("hash") or computed
        print(f"[OK] Verified block: {block_path.name}")

    return True

if __name__ == "__main__":
    import sys
    success = verify_chain(Path(__file__).parent / "chain")
    if not success:
        sys.exit(1)
