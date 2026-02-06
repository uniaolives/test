# scripts/metatron_deploy.py
import asyncio
import sys
import os

# Adiciona o diretório atual ao path
sys.path.append(os.getcwd())

from cosmos.metatron import MetatronDistributor

async def main():
    print("🏛️ INICIANDO PROTOCOLO METATRON")
    distributor = MetatronDistributor()

    print("\n[FASE 1] Cristalização Orbital S (Alpha - 12 nós)")
    await distributor.run_crystallization('S')

    print("\n[FASE 2] Cristalização Orbital P (Beta - 60 nós)")
    await distributor.run_crystallization('P')

    print("\n📊 RESUMO DA OPERAÇÃO")
    print(f"Total de nós cristalizados: {distributor.completed}")
    print(f"Frequências ativas: {distributor.frequency_monitor}")
    print("Catedral Fermiônica agora respira em harmonia quântica. o<>o")

if __name__ == "__main__":
    asyncio.run(main())
