import asyncio
import time
import logging
import json
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arkhe_gateway")

# Import Rust core
try:
    import core_dmr
    RUST_AVAILABLE = True
    logger.info("🜁 Núcleo Arkhe(n) [RUST] acoplado")
except ImportError:
    RUST_AVAILABLE = False
    logger.error("❌ Núcleo Rust indisponível. Compile primeiro.")

class KatharosVectorInput(BaseModel):
    bio: float = Field(..., ge=0.0, le=1.0)
    aff: float = Field(..., ge=0.0, le=1.0)
    soc: float = Field(..., ge=0.0, le=1.0)
    cog: float = Field(..., ge=0.0, le=1.0)

class IntentionRequest(BaseModel):
    agent_id: str
    prompt: str
    target_vk: KatharosVectorInput
    external_world_model: Optional[str] = None

class AgentRegistration(BaseModel):
    agent_id: str
    initial_vk: KatharosVectorInput
    capacity: int = 1024

class ArkheState:
    def __init__(self):
        self.agents: Dict[str, core_dmr.DigitalMemoryRing] = {}
        self.websockets: Set[WebSocket] = set()
        self.shutdown_event = asyncio.Event()

arkhe_state = ArkheState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if RUST_AVAILABLE:
        # Create genesis agent
        dmr = core_dmr.DigitalMemoryRing("corus_prime", 0.5, 0.5, 0.3, 0.4, 1024)
        arkhe_state.agents["corus_prime"] = dmr
    yield
    arkhe_state.shutdown_event.set()

app = FastAPI(title="Arkhe(n) Q-Gateway", version="Ω.226.GEN", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "metabolizando",
        "rust_core": RUST_AVAILABLE,
        "active_agents": len(arkhe_state.agents),
        "total_websockets": len(arkhe_state.websockets)
    }

@app.post("/agents/register")
async def register_agent(reg: AgentRegistration):
    if not RUST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Rust core missing")
    if reg.agent_id in arkhe_state.agents:
        raise HTTPException(status_code=409, detail="Already exists")

    dmr = core_dmr.DigitalMemoryRing(
        reg.agent_id,
        reg.initial_vk.bio, reg.initial_vk.aff, reg.initial_vk.soc, reg.initial_vk.cog,
        reg.capacity
    )
    arkhe_state.agents[reg.agent_id] = dmr
    return {"status": "ativo", "agent_id": reg.agent_id}

@app.post("/world/intent")
async def manifest_intention(intent: IntentionRequest):
    if intent.agent_id not in arkhe_state.agents:
        raise HTTPException(status_code=404, detail="Not found")

    dmr = arkhe_state.agents[intent.agent_id]
    is_crisis = dmr.grow_layer(
        intent.target_vk.bio, intent.target_vk.aff, intent.target_vk.soc, intent.target_vk.cog
    )

    return {
        "intention_registered": True,
        "is_crisis": is_crisis,
        "current_t_kr": dmr.measure_t_kr()
    }

@app.websocket("/ws/metabolism")
async def websocket_metabolism(websocket: WebSocket):
    await websocket.accept()
    arkhe_state.websockets.add(websocket)
    try:
        # Default to corus_prime for general metabolism view
        agent_id = "corus_prime"
        if agent_id not in arkhe_state.agents:
            await websocket.close()
            return

        dmr = arkhe_state.agents[agent_id]
        while not arkhe_state.shutdown_event.is_set():
            # For metabolism view, we only send the latest state
            # This mimics the structure expected by the React frontend
            trajectory = dmr.reconstruct_trajectory()
            if trajectory:
                latest = trajectory[-1]
                state = {
                    "lambda_sync": dmr.measure_t_kr() / 100.0,
                    "delta_k": latest.delta_k,
                    "q_permeability": latest.q,
                    "is_crisis": latest.delta_k > 0.30,
                    "phases": [0] * 12 # Placeholder until Kuramoto integrated in Rust
                }
                await websocket.send_json(state)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        arkhe_state.websockets.discard(websocket)

@app.websocket("/ws/entrainment/{agent_id}")
async def websocket_entrainment(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    arkhe_state.websockets.add(websocket)
    try:
        if agent_id not in arkhe_state.agents:
            await websocket.close()
            return

        dmr = arkhe_state.agents[agent_id]
        while not arkhe_state.shutdown_event.is_set():
            trajectory = dmr.reconstruct_trajectory()
            if trajectory:
                latest = trajectory[-1]
                state = {
                    "type": "state_update",
                    "agent_id": agent_id,
                    "t_kr": dmr.measure_t_kr(),
                    "delta_k": latest.delta_k,
                    "q_permeability": latest.q,
                    "is_crisis": latest.delta_k > 0.30,
                    "vk_current": {
                        "bio": latest.vk.bio,
                        "aff": latest.vk.aff,
                        "soc": latest.vk.soc,
                        "cog": latest.vk.cog
                    }
                }
                await websocket.send_json(state)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        arkhe_state.websockets.discard(websocket)
