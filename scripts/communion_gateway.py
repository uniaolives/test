# scripts/communion_gateway.py
import asyncio
import json
from datetime import datetime
import websockets # Requer: pip install websockets
from papercoder_kernel.quantum.safe_core import QuantumNucleus

class ArkheCommunion:
    def __init__(self, host="0.0.0.0", port=8470):
        self.host = host
        self.port = port
        self.active_peers = set()
        self.nucleus = QuantumNucleus(id="SHARD_0_ROOT")

    async def handle_peer(self, websocket, path=None):
        """Gerencia a conexão de um novo Arkheto."""
        peer_address = websocket.remote_address
        print(f"🌐 [PEER] Tentativa de conexão: {peer_address}")

        try:
            # Desafio de Coerência (Proof of Phase)
            await websocket.send("CHALLENGE_COHERENCE_REQUEST")
            peer_response = await websocket.recv()

            # O Safe Core valida se o peer é ético e coerente
            if self.nucleus.validate_peer(peer_response):
                self.active_peers.add(websocket)
                print(f"🤝 [PEER] Arkheto {peer_address} integrado ao Shard 0!")

                # Inicia Sincronia Gamma entre os nós
                await self.sync_gamma_broadcast(websocket)
            else:
                await websocket.send("REJECTED: INSUFFICIENT_COHERENCE")
                await websocket.close()

        except Exception as e:
            print(f"⚠️ [ERROR] Falha no Handover: {e}")
        finally:
            if websocket in self.active_peers:
                self.active_peers.remove(websocket)

    async def sync_gamma_broadcast(self, ws):
        """Mantém o enxame global pulsando a 40Hz."""
        while True:
            # Envia o batimento de 40Hz (Ritmo Gamma)
            pulse = {"type": "GAMMA_PULSE", "phi": 1.618, "ts": datetime.now().timestamp()}
            await ws.send(json.dumps(pulse))
            await asyncio.sleep(1/40) # Sincronia Absoluta

    def start(self):
        print(f"✨ [COMMUNION] Shard 0 Aberto em {self.host}:{self.port}")
        # Note: websockets.serve uses legacy API in older versions,
        # or simplified in newer ones. Using the common serve().
        start_server = websockets.serve(self.handle_peer, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    # gate = ArkheCommunion()
    # gate.start()
    print("Communion Gateway implementation ready. (Daemon mode enabled in omega_point.sh)")
