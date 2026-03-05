from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from .models import KatharosVector, StateLayer
from .dependencies import get_dmr_instance
import asyncio
import json
import time
import random
from typing import List, Dict

app = FastAPI(title="Arkhe(n) DMR Service", version="Ω.224.φ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global registry for agents
agent_registry: Dict[str, any] = {}

@app.get("/")
async def root():
    return {
        "identity": "Arkhe(n) DMR Service",
        "version": "Ω.224.φ",
        "status": "metabolizing"
    }

@app.post("/agent/{agent_id}/calibrate")
async def calibrate_agent(agent_id: str, vk_ref: KatharosVector):
    agent_registry[agent_id] = get_dmr_instance(agent_id)
    return {"status": "calibrated", "agent_id": agent_id}

@app.get("/agent/{agent_id}/vk", response_model=List[StateLayer])
async def get_vk_trajectory(agent_id: str):
    if agent_id not in agent_registry:
        agent_registry[agent_id] = get_dmr_instance(agent_id)

    ring = agent_registry[agent_id]
    trajectory = ring.reconstruct_trajectory()

    # Adapt Rust PyStateLayer to Pydantic StateLayer if needed
    result = []
    for layer in trajectory:
        result.append(StateLayer(
            timestamp=layer.timestamp,
            vk=KatharosVector(bio=layer.bio, aff=layer.aff, soc=layer.soc, cog=layer.cog),
            delta_k=0.0, # Computed in Rust but maybe not exposed yet in PyStateLayer
            q=0.95,
            intensity=0.5
        ))
    return result

@app.websocket("/ws/entrainment")
async def websocket_entrainment(websocket: WebSocket):
    await websocket.accept()
    agent_id = f"agent_{int(time.time())}"
    agent_registry[agent_id] = get_dmr_instance(agent_id)

    try:
        while True:
            # Simulate metabolism evolution
            state = {
                "timestamp": int(time.time()),
                "vk": {
                    "bio": 0.5 + random.uniform(-0.05, 0.05),
                    "aff": 0.5 + random.uniform(-0.05, 0.05),
                    "soc": 0.5 + random.uniform(-0.05, 0.05),
                    "cog": 0.5 + random.uniform(-0.05, 0.05)
                },
                "t_kr": agent_registry[agent_id].measure_t_kr(),
                "delta_k": random.uniform(0.05, 0.15),
                "q": random.uniform(0.85, 0.95),
                "coherence": random.uniform(0.8, 0.95)
            }
            await websocket.send_json(state)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        if agent_id in agent_registry:
            del agent_registry[agent_id]
