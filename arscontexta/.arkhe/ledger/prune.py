# .arkhe/ledger/prune.py
import os
import shutil
from pathlib import Path

def prune_ledger(chain_dir: Path, archive_dir: Path, keep_last=100):
    """
    Arquivamento de blocos antigos mantendo os últimos N.
    """
    blocks = sorted(list(chain_dir.glob("*.json")))
    if len(blocks) <= keep_last:
        return

    to_prune = blocks[:-keep_last]
    archive_dir.mkdir(exist_ok=True)

    for block in to_prune:
        shutil.move(str(block), str(archive_dir / block.name))
        print(f"[LEDGER] Pruned and archived: {block.name}")

if __name__ == "__main__":
    prune_ledger(
        Path("arscontexta/.arkhe/ledger/chain"),
        Path("arscontexta/.arkhe/ledger/archive")
    )
