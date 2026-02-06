# app/sentinel_webhook.py
"""
Centralized Webhook for system events and Rosehip neural signatures.
"""
from fastapi import FastAPI, Request, BackgroundTasks
import hmac
import hashlib
import json
import os
import sys

# Cosmopsychia integration
sys.path.append(os.getcwd())
from cosmos.ontological import OntologicalKernel

app = FastAPI(title="Cosmopsychia Sentinel Webhook")
kernel = OntologicalKernel()

@app.post("/webhook/signature")
async def handle_neural_signature(request: Request):
    payload = await request.json()
    entropy = payload.get("entropy", 0.0)
    commit_hash = payload.get("hash", "unknown")

    print(f"🌹 [WEBHOOK] Received Neural Signature for {commit_hash}: Entropy={entropy}")

    # Validation against ontological kernel
    try:
        coherence = 1.0 - (entropy / 20.0) # Map entropy to coherence
        kernel.validate_layer_coherence("computational", max(0.5, coherence))
        status = "Aligned"
    except Exception as e:
        status = "Geometric Dissonance"

    return {
        "status": status,
        "signature_recorded": True,
        "system_health": kernel.get_system_health()
    }

@app.get("/cosmic_health")
async def get_health():
    # In a real system, this would query tim_vm
    return {
        "status": "harmonious",
        "dilation": 1.02,
        "entropy_avg": 5.4,
        "rosehip_active": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
