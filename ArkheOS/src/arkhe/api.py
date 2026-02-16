# arkhe/api.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
import time
from .consensus_syzygy import ProofOfSyzygy
from .neuro_mapping import NeuroMappingProcessor
from .recalibration import RecalibrationProtocol

app = FastAPI(title="Arkhe(N) Sovereign API", version="5.0.0")

# Mock State
class SystemState:
    def __init__(self):
        self.darvo_params = {"key_lifetime": 24.0, "attestation_required": True, "threshold": 0.8}
        self.nodes = [
            {"id": "alpha", "name": "α (Primordial)", "coherence": 1.0, "status": "healthy"},
            {"id": "beta", "name": "β (Estrutural)", "coherence": 0.94, "status": "healthy"},
            {"id": "gamma", "name": "γ (Temporal)", "coherence": 0.88, "status": "warning"}
        ]
        self.handovers = [
            {"id": "H_001", "timestamp": time.time(), "type": "GENESIS", "status": "SUCCESS"},
            {"id": "H_002", "timestamp": time.time() + 1, "type": "QKD_ROTATION", "status": "SUCCESS"}
        ]
        self.results_dir = "fsl_sim_results"

state = SystemState()

class DarvoConfig(BaseModel):
    key_lifetime: float
    attestation_required: bool
    threshold: float

@app.get("/api/status")
async def get_status():
    return {
        "nodes": state.nodes,
        "global_coherence": sum(n['coherence'] for n in state.nodes) / len(state.nodes),
        "darvo_active": True
    }

@app.get("/api/logs/handovers")
async def get_handovers():
    return state.handovers

@app.post("/api/config/darvo")
async def update_darvo(config: DarvoConfig):
    state.darvo_params.update(config.dict())
    return {"message": "Configuração Darvo atualizada", "current": state.darvo_params}

@app.post("/api/consensus/validate")
async def validate_handover(proposal_id: str):
    # Usar os estados reais dos nós para o consenso
    posyz = ProofOfSyzygy(
        alpha_c=state.nodes[0]['coherence'],
        beta_c=state.nodes[1]['coherence'],
        gamma_c=state.nodes[2]['coherence']
    )
    result = posyz.validate_handover(proposal_id)
    return result

@app.get("/api/neuro/stats")
async def get_neuro_stats():
    processor = NeuroMappingProcessor(state.results_dir)
    return processor.process_ledgers()

@app.get("/api/recalibration")
async def get_recalibration(nexus: str = "MAR_26"):
    processor = NeuroMappingProcessor(state.results_dir)
    neuro_report = processor.process_ledgers()
    recal = RecalibrationProtocol(neuro_report)
    return recal.generate_plan(nexus)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
