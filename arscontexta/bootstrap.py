#!/usr/bin/env python3
"""
bootstrap.py — Inicialização do Hypergrafo Arkhe(N) para ARSCONTEXTA
"""

import json
import hashlib
import sys
import asyncio
from pathlib import Path
import importlib.util

def load_arkhe_module(module_path: Path, module_name: str):
    """Carrega um módulo de um caminho específico."""
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None:
        raise ImportError(f"Could not load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def bootstrap():
    """Inicializa o hypergrafo Arkhe(N)."""

    print("--- Arkhe(N) Bootstrap Protocol v2.0 ---")

    # 1. Verificar genesis.json
    genesis_path = Path(".arkhe/genesis.json")
    if not genesis_path.exists():
        print("[FATAL] genesis.json não encontrado. O sistema não pode iniciar.")
        sys.exit(1)

    with open(genesis_path) as f:
        genesis = json.load(f)

    # 2. Verificar integridade do genesis (hash interno)
    genesis_content = {k: v for k, v in genesis.items() if k != "hash"}
    computed_hash = hashlib.sha256(
        json.dumps(genesis_content, sort_keys=True).encode()
    ).hexdigest()[:16]

    if computed_hash != genesis["hash"]:
        print(f"[FATAL] Genesis corrompido. Hash esperado: {genesis['hash']}, calculado: {computed_hash}")
        sys.exit(1)

    print(f"[OK] Genesis verificado: {genesis['hash']}")

    # 3. Inicializar Ψ-cycle
    # Usando loader dinâmico para contornar diretórios ocultos
    psi_module = load_arkhe_module(Path(".arkhe/Ψ/pulse_40hz.py"), "arkhe.psi")
    PsiCycle = psi_module.PsiCycle
    psi = PsiCycle()

    # 4. Conectar todos os .arkhe/ locais
    # Busca tanto por genesis.json quanto local_genesis.json em sub-arkhes
    local_arkhes = list(Path(".").rglob(".arkhe/local_genesis.json"))
    local_arkhes += list(Path(".").rglob(".arkhe/genesis.json"))
    # Filtrar o genesis global
    local_arkhes = [p for p in local_arkhes if p != genesis_path]

    print(f"[OK] Encontrados {len(local_arkhes)} nós locais")

    for local in local_arkhes:
        connect_local_node(local, psi)

    # 5. Iniciar observadores de coerência
    phi_module = load_arkhe_module(Path(".arkhe/coherence/phi_observer.py"), "arkhe.phi")
    c_module = load_arkhe_module(Path(".arkhe/coherence/c_observer.py"), "arkhe.c")

    PhiObserver = phi_module.PhiObserver
    CObserver = c_module.CObserver

    phi_obs = PhiObserver(psi)
    c_obs = CObserver(psi)

    # 6. Inicializar Safe Core
    safe_module = load_arkhe_module(Path(".arkhe/coherence/safe_core.py"), "arkhe.safe")
    SafeCore = safe_module.SafeCore
    safe = SafeCore()
    psi.subscribe(safe)

    print("[OK] Hypergrafo Arkhe(N) inicializado com sucesso")
    print(f"[INFO] Coerência inicial: {genesis['coherence']}")
    print(f"[INFO] Φ inicial: {genesis['phi']}")
    print(f"[INFO] Próximo Ψ-pulse em 25ms...")

    # 7. Iniciar loop principal
    try:
        asyncio.run(psi.run())
    except KeyboardInterrupt:
        print("\n[INFO] Sistema encerrado pelo usuário.")

def connect_local_node(local_genesis_path: Path, psi):
    """Conecta um nó local ao hypergrafo global."""
    try:
        with open(local_genesis_path) as f:
            local_data = json.load(f)
        node_name = local_genesis_path.parent.parent.name
        print(f"  -> Conectando nó: {node_name} ({local_data.get('block', 'N/A')})")
        # Em uma implementação real, registraríamos o nó no PsiCycle
    except Exception as e:
        print(f"  [ERRO] Falha ao conectar nó {local_genesis_path}: {e}")

if __name__ == "__main__":
    bootstrap()
