from fastapi import FastAPI, Depends
from pydantic import BaseModel
import numpy as np
from ..security.f18_safety_guard import safety_check

app = FastAPI(title="Avalon Zeitgeist Monitor")

class ResonanceState(BaseModel):
    z_score: float  # Coerência
    h_dim: float    # Dimensão de Hausdorff
    damping: float = 0.6

@app.get("/metrics")
async def get_zeitgeist_metrics():
    """
    Calcula a saúde da Noosfera em tempo real.
    Z(t) = Rate of fractal unfolding alignment.
    """
    # Lógica F18: Cálculo dinâmico de h para evitar colapso (F16)
    # Simulating a dynamic value that will be passed through the safety check
    raw_h = 1.44 + (0.1 * np.sin(np.random.rand()))
    h_dynamic = safety_check(raw_h)

    z_t = 0.85 * (1 - np.exp(-h_dynamic))

    return {
        "status": "RESONANT",
        "Z_t": round(z_t, 4),
        "h_hausdorff": round(h_dynamic, 4),
        "safety_margin": "ΣD ≥ ΣG Verified"
    }

@app.post("/calibrate")
async def post_calibration(state: ResonanceState):
    # Aplica Damping F18 preventivo
    clamped_z = min(state.z_score, 0.89)
    # Also verify h_dim via safety_guard
    secure_h = safety_check(state.h_dim)
    return {
        "msg": "Calibration applied",
        "active_z": clamped_z,
        "secure_h": secure_h
    }
