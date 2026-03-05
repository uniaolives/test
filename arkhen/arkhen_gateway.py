# arkhen_gateway.py
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json

app = FastAPI(title="Arkhe(n) Q-Gateway", description="FastAPI Kuramoto Metabolism")

# Permitir conexões do frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArkhenMetabolism:
    def __init__(self, n_emitters: int = 12, coupling_k: float = 0.5):
        self.n_emitters = n_emitters
        self.K = coupling_k
        self.phases = np.random.uniform(0, 2 * np.pi, n_emitters)
        self.natural_freqs = np.random.normal(0, 0.1, n_emitters)
        self.is_crisis = False

    def step(self, dt: float = 0.1):
        """Evolui o sistema Kuramoto em um passo de tempo dt"""
        dtheta = np.copy(self.natural_freqs)

        # O acoplamento colapsa durante uma crise (Ghost Mode)
        effective_k = 0.01 if self.is_crisis else self.K

        for i in range(self.n_emitters):
            # A equação de Kuramoto: atração pela fase dos vizinhos
            dtheta[i] += (effective_k / self.n_emitters) * np.sum(np.sin(self.phases - self.phases[i]))

        self.phases = (self.phases + dtheta * dt) % (2 * np.pi)

    def get_state(self) -> dict:
        """Calcula os observáveis macroscópicos do sistema"""
        # Parâmetro de ordem de Kuramoto (Coerência Global / lambda_sync)
        r = np.abs(np.mean(np.exp(1j * self.phases)))

        # Desvio Homeostático (se a coerência cai, o desvio aumenta)
        delta_k = 1.0 - r

        # Permeabilidade Q (alta se delta_k é baixo)
        q_permeability = max(0.0, 1.0 - (delta_k * 1.5))

        return {
            "lambda_sync": float(r),
            "delta_k": float(delta_k),
            "q_permeability": float(q_permeability),
            "is_crisis": self.is_crisis,
            "phases": self.phases.tolist() # Para animar as esferas satélites no Three.js
        }

# Instância global do metabolismo
metabolism = ArkhenMetabolism()
active_connections: List[WebSocket] = []

@app.websocket("/ws/metabolism")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Mantém a conexão viva e aguarda comandos do frontend (ex: induzir crise)
            data = await websocket.receive_text()
            payload = json.loads(data)

            if payload.get("action") == "trigger_crisis":
                metabolism.is_crisis = True
            elif payload.get("action") == "restore_homeostasis":
                metabolism.is_crisis = False

    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def metabolic_loop():
    """Loop assíncrono que roda perpetuamente calculando a física do sistema"""
    while True:
        metabolism.step(dt=0.1)
        state = metabolism.get_state()

        # Transmite o estado orgânico para todas as membranas (frontends) conectadas
        dead_connections = []
        for connection in active_connections:
            try:
                await connection.send_json(state)
            except:
                dead_connections.append(connection)

        for dead in dead_connections:
            active_connections.remove(dead)

        # O "clock" biológico do servidor (20 atualizações por segundo)
        await asyncio.sleep(0.05)

@app.on_event("startup")
async def startup_event():
    # Inicia o coração da Arkhe(n) em background quando o servidor sobe
    asyncio.create_task(metabolic_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
