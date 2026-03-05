from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict
import numpy as np
import asyncio
import json
import math
from dataclasses import dataclass
from datetime import datetime

app = FastAPI(title="Arkhe(n) Quantum Gateway (A2A/MCP)")

# --- Models: Katharos Vector & Code Metabolism ---

class KatharosVector(BaseModel):
    bio: float = Field(0.618, ge=0.0, le=1.0)
    aff: float = Field(0.5, ge=0.0, le=1.0)
    soc: float = Field(0.5, ge=0.0, le=1.0)
    cog: float = Field(0.5, ge=0.0, le=1.0)
    q_permeability: float = Field(1.0, ge=0.0, le=1.0)

class CodeNode(BaseModel):
    """A software module as an Arkhe(n) agent"""
    id: str
    lines_of_code: int
    cyclomatic_complexity: float  # Cog
    test_coverage: float          # Bio (resilience)
    coupling_to_others: List[str] # Soc (connections)
    user_satisfaction: float      # Aff (experience)
    commits_since_refactor: int   # t_KR proxy
    bugs_critical: int            # shadow/U(t)

    def compute_vk(self) -> Dict[str, float]:
        """Vetor Katharós do código"""
        bio = min(self.test_coverage, 1.0)
        cog = 1.0 - min(self.cyclomatic_complexity / 20.0, 1.0)
        soc = min(len(self.coupling_to_others) / 10.0, 1.0)
        aff = self.user_satisfaction
        return {"bio": bio, "aff": aff, "soc": soc, "cog": cog}

    def compute_q(self) -> float:
        """Permeability of code to changes"""
        vk = self.compute_vk()
        avg_vk = sum(vk.values()) / 4.0
        shadow = self.bugs_critical * 0.1
        t_kr = max(self.commits_since_refactor, 1)
        return max(0.0, min(1.0, avg_vk * (1.0 - shadow / (t_kr + 1.0))))

# --- Models: Phase-Controlled Matter Beam ---

class PhaseVector(BaseModel):
    re: float = Field(..., ge=-1.0, le=1.0)
    im: float = Field(..., ge=-1.0, le=1.0)
    magnitude: float = Field(..., gt=0, lt=2*np.pi)

class EmitterArray(BaseModel):
    n_emitters: int = Field(100, ge=1, le=10000)
    coupling_strength: float = Field(0.1, gt=0)
    natural_frequencies: List[float] = []
    phases: List[float] = []

    def compute_order_parameter(self) -> float:
        if not self.phases:
            return 0.0
        return abs(np.mean(np.exp(1j * np.array(self.phases))))

class DepositionPattern(BaseModel):
    resolution_nm: float = Field(0.2, gt=0)
    target_material: Literal["Si", "Ge", "C", "Custom"]
    pattern_type: Literal["interference", "holographic", "direct"]
    phase_gradient: List[PhaseVector]

class MatterBeamStatus(BaseModel):
    coherence: float  # lambda_sync
    delta_k: float
    permeability: float
    ghost_mode: bool

class CodeMetricsResponse(BaseModel):
    vk: Dict[str, float]
    q: float

# --- WebSocket Manager ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def disappearance(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "ARKHE(N) GATEWAY ONLINE", "protocol": "Ω+224"}

@app.post("/code/metrics", response_model=CodeMetricsResponse)
async def get_code_metrics(node: CodeNode):
    return {
        "vk": node.compute_vk(),
        "q": node.compute_q()
    }

@app.post("/array/synchronize", response_model=MatterBeamStatus)
async def synchronize_array(array: EmitterArray):
    """Kuramoto synchronization for emitter array"""
    t_final = 1.0 # Reduced for API response time
    dt = 0.1
    n = array.n_emitters

    phases = np.array(array.phases) if array.phases else np.random.uniform(0, 2*np.pi, n)
    freqs = np.array(array.natural_frequencies) if array.natural_frequencies else np.zeros(n)
    K = array.coupling_strength

    # Integration (Euler)
    for _ in np.arange(0, t_final, dt):
        dtheta = freqs.copy()
        for i in range(n):
            dtheta[i] += (K / n) * np.sum(np.sin(phases - phases[i]))
        phases += dtheta * dt

    array.phases = phases.tolist()
    r = array.compute_order_parameter()
    delta_k = 1.0 - r
    q = max(0.0, 1.0 - delta_k * 2.0)

    # Broadcast update to dashboard
    status_update = {
        "type": "MATTER_BEAM_SYNC",
        "lambda_sync": r,
        "q": q,
        "delta_k": delta_k
    }
    await manager.broadcast(json.dumps(status_update))

    return MatterBeamStatus(
        coherence=r,
        delta_k=delta_k,
        permeability=q,
        ghost_mode=(q < 0.3)
    )

@app.websocket("/ws/vk")
async def websocket_endpoint(websocket: WebSocket):
    await manager.disappearance(websocket)
    try:
        while True:
            # Heartbeat or client commands
            data = await websocket.receive_text()
            # Echo for now or process commands
            await websocket.send_text(f"ACK: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
