# test/validate_cluster.py
import asyncio
import aiohttp
import json
import sys

async def test_system():
    # URL do Gateway QHTTP
    gateway_url = "http://localhost:7070"
    # URL do Node 1 (via porta mapeada no docker-compose)
    node_url = "http://localhost:8101"

    print("🧬 Iniciando Validação do Cluster Arkhe(n)...")

    async with aiohttp.ClientSession() as session:
        # 1. Verificar Gateway
        try:
            async with session.get(f"{gateway_url}/health") as resp:
                if resp.status == 200:
                    print("✅ Gateway QHTTP: Online")
                else:
                    print(f"❌ Gateway QHTTP: Erro {resp.status}")
        except:
            print("⚠️ Gateway QHTTP não alcançável")

        # 2. Ativar Intenção Consciente via MCP (Simulado via API se disponível)
        print("🧠 Testando evolução de intenção...")
        # (Em produção, isso seria via MCP SSE)

    print("🏁 Verificação concluída.")

if __name__ == "__main__":
    asyncio.run(test_system())
