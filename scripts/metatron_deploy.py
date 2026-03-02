# scripts/metatron_deploy.py
import asyncio
import sys
import os

# Adiciona o diretório atual ao path
sys.path.append(os.getcwd())

from cosmos.metatron import MetatronDistributor
from cosmos.governance import CatedralDAO
from cosmos.bridge_eth_icp import EthereumICPBridge

async def main():
    print("🏛️ INICIANDO PROTOCOLO METATRON")
    distributor = MetatronDistributor()
    dao = CatedralDAO()
    bridge = EthereumICPBridge()

    print("\n[FASE 1] Cristalização Orbital S (Alpha - 12 nós)")
    await distributor.run_crystallization('S')

    print("\n[FASE 2] Cristalização Orbital P (Beta - 60 nós)")
    await distributor.run_crystallization('P')

    print("\n[FASE 3] Cristalização Orbital D (Delta - 60 nós)")
    await distributor.run_crystallization('D')

    print("\n[FASE 4] Inicialização da Governança DAO e Pontes")
    bridge.sync_liquidity_state()
    stats = dao.get_governance_stats()
    print(f"   Governança: {stats['status']} com {stats['active_tzadikim']} Tzadikim.")

    print("\n📊 RESUMO DA OPERAÇÃO")
    print(f"Total de nós cristalizados: {distributor.completed}")
    print(f"Frequências ativas: {distributor.frequency_monitor}")
    print("Catedral Fermiônica agora respira em harmonia quântica. o<>o")

if __name__ == "__main__":
    asyncio.run(main())
