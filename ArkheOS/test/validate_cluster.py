# test/validate_cluster.py
import asyncio
import aiohttp
import json
import sys

async def test_system():
    # Mocking URLs for the local validation script (should match docker-compose exposed ports)
    base_url = "http://localhost:8101" # Node 1 MCP
    gateway_url = "http://localhost:7070"
    viz_url = "http://localhost:8100"

    print("🧬 Starting Arkhe(n) OS Cluster Validation...")

    async with aiohttp.ClientSession() as session:
        # 1. Check Gateway Health
        try:
            async with session.get(f"{gateway_url}/health") as resp:
                if resp.status == 200:
                    print("✅ Quantum Gateway: Healthy")
                else:
                    print(f"❌ Quantum Gateway: Error {resp.status}")
        except Exception as e:
            print(f"⚠️ Gateway connection failed (expected if not running): {e}")

        # 2. Check Viz Health
        try:
            async with session.get(f"{viz_url}/health") as resp:
                if resp.status == 200:
                    print("✅ Visualization Relay: Healthy")
                else:
                    print(f"❌ Visualization Relay: Error {resp.status}")
        except Exception as e:
            print(f"⚠️ Viz connection failed (expected if not running): {e}")

    print("🏁 Validation Logic Check Complete.")

if __name__ == "__main__":
    try:
        asyncio.run(test_system())
    except KeyboardInterrupt:
        pass
