from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import KatharosVector, StateLayer
from .dependencies import get_dmr_instance
from .hyperclaw.loops import HyperClawOrchestrator, ContextFrame
from .geoloc.poloc import BftPoLoc
from .physics.simulators import QuantumSimulator
from .physics.triggers import ArkheTrigger
from .blockchain.satoshi import verify_satoshi_temporal
from .quantum.noether import QHTTPNoetherBridge
from contextlib import asynccontextmanager
import asyncio
import json
import time
import random
from typing import List, Dict

# Global registry for agents
agent_registry: Dict[str, any] = {}
hyperclaw_orchestrator = HyperClawOrchestrator()
geoloc_verifier = BftPoLoc(beta=0.2)
quantum_sim = QuantumSimulator()
lhc_trigger = ArkheTrigger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup HyperClaw for a default frame
    frame_id = "default_frame"
    hyperclaw_orchestrator.frames[frame_id] = ContextFrame()
    await hyperclaw_orchestrator.start(frame_id)
    yield
    hyperclaw_orchestrator.running = False

app = FastAPI(
    title="Arkhe(n) DMR Service",
    version="Ω.224.φ",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            delta_k=0.0,
            q=0.95,
            intensity=0.5
        ))
    return result

@app.get("/hyperclaw/templates")
async def get_hyperclaw_templates():
    from .hyperclaw.templates import BIOTECH_TEMPLATES
    return BIOTECH_TEMPLATES

@app.post("/hyperclaw/spawn/{template_id}")
async def spawn_hyperclaw_frame(template_id: str, frame_id: str):
    await hyperclaw_orchestrator.spawn_templated_frame(frame_id, template_id)
    return {"status": "spawned", "frame_id": frame_id, "template_id": template_id}

@app.get("/hyperclaw/frame/{frame_id}")
async def get_hyperclaw_frame(frame_id: str):
    frame = hyperclaw_orchestrator.frames.get(frame_id)
    if not frame:
        return {"error": "Frame not found"}
    return {
        "mode": frame.mode.value,
        "goals": frame.goals,
        "budget": frame.budget
    }

@app.post("/physics/lhc/analyze")
async def analyze_lhc_event(jets: List[Dict]):
    return lhc_trigger.evaluate_event(jets)

@app.get("/physics/quantum/decoherence")
async def get_orbital_decoherence(h: float = 400e3):
    return {"tau_c": quantum_sim.orbital_decoherence(h)}

@app.post("/blockchain/satoshi/verify")
async def verify_satoshi(blocks: List[Dict]):
    return await verify_satoshi_temporal(blocks)

@app.post("/geoloc/verify")
async def verify_location(agent_id: str, lat: float, lon: float, measurements: List[Dict]):
    """
    Verifies location using BFT-PoLoc.
    measurements: List of {'lat': float, 'lon': float, 'rtt': float}
    """
    result = geoloc_verifier.verify(lat, lon, measurements)

    if agent_id in agent_registry:
        ring = agent_registry[agent_id]
        # High uncertainty increases delta_k (simulated via growth)
        if result["is_valid"]:
            ring.grow_layer(0.5, 0.5, 0.5, 0.5, 0.95)
        else:
            ring.grow_layer(0.7, 0.7, 0.7, 0.7, 0.3)

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
                "coherence": random.uniform(0.8, 0.95),
                "hyperclaw_mode": hyperclaw_orchestrator.frames["default_frame"].mode.value if "default_frame" in hyperclaw_orchestrator.frames else "N/A"
            }
            await websocket.send_json(state)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        if agent_id in agent_registry:
            del agent_registry[agent_id]
