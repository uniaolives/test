from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .models_pydantic import KatharosVector, StateLayer, SystemState
from .dependencies import get_dmr_instance, RUST_AVAILABLE
from .database import get_db, init_db, check_db_health, get_db_stats
from .repositories.dmr_repository import DMRRepository
from .hyperclaw.loops import HyperClawOrchestrator, ContextFrame
from .geoloc.poloc import BftPoLoc
from .physics.simulators import QuantumSimulator
from .physics.triggers import ArkheTrigger
from .physics.arkhe_s2_lhc import LHCDataLoader, ArkheLHCTrigger, ArkheLHCAnalysis
from .blockchain.satoshi import verify_satoshi_temporal
from .quantum.qiskit_circuits import (
    novikov_loop_circuit, novikov_loop_kraus, trefoil_knot_circuit,
    detect_wave_cloud_nucleation, QiskitInterface
)
from qiskit import qasm2
from .knowledge.google_scanner import SemanticMiner
from .monitoring.listener import RealityListener
from .middleware.constitution import ConstitutionalGuard
from contextlib import asynccontextmanager
import asyncio
import json
import time
import random
import os
from typing import List, Dict, Any

# Operational Mode Detection
DATABASE_AVAILABLE = check_db_health()
OPERATIONAL_MODE = os.getenv("OPERATIONAL_MODE", "FULL")

if OPERATIONAL_MODE == "FULL":
    if not DATABASE_AVAILABLE and not RUST_AVAILABLE:
        OPERATIONAL_MODE = "MOCK"
    elif not DATABASE_AVAILABLE:
        OPERATIONAL_MODE = "MEMORY ONLY"
    elif not RUST_AVAILABLE:
        OPERATIONAL_MODE = "DATABASE ONLY"

# Global registry for agents (legacy/fallback)
agent_registry: Dict[str, any] = {}
te_estimators: Dict[str, any] = {}
hydraulic_engines: Dict[str, any] = {}

hyperclaw_orchestrator = HyperClawOrchestrator()
geoloc_verifier = BftPoLoc(beta=0.2)
quantum_sim = QuantumSimulator()
lhc_trigger = ArkheTrigger()
s2_trigger = ArkheLHCTrigger()
qiskit_iface = QiskitInterface()
reality_listener = RealityListener()
system_state = SystemState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database if available
    if OPERATIONAL_MODE in ["FULL", "DATABASE ONLY"]:
        try:
            init_db()
            print("[INFO] Database initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Database initialization failed: {e}")

    # Startup HyperClaw for a default frame
    frame_id = "default_frame"
    hyperclaw_orchestrator.frames[frame_id] = ContextFrame()
    await hyperclaw_orchestrator.start(frame_id)
    # Start Reality Listener
    reality_listener.start()
    yield
    hyperclaw_orchestrator.running = False
    reality_listener.stop()

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
app.add_middleware(ConstitutionalGuard)

@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    if OPERATIONAL_MODE in ["FULL", "DATABASE ONLY"]:
        try:
            db = next(get_db())
            repo = DMRRepository(db)
            # Constante Elena H calculation (symbolic/proxy)
            h_val = random.uniform(0.1, 0.9)
            repo.log_constitutional_action(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                h_value=h_val,
                processing_time_ms=process_time,
                status="VALIDATED" if response.status_code < 400 else "REJECTED"
            )
        except Exception as e:
            print(f"Audit log failed: {e}")

    return response

@app.get("/")
async def root():
    return {
        "identity": "Arkhe(n) DMR Service",
        "version": "Ω.224.φ",
        "status": "metabolizing",
        "mode": OPERATIONAL_MODE,
        "database": "online" if DATABASE_AVAILABLE else "offline",
        "rust_dmr": "available" if RUST_AVAILABLE else "mocked"
    }

@app.get("/database/health")
async def get_db_health():
    is_healthy = check_db_health()
    return {"status": "healthy" if is_healthy else "unhealthy", "mode": OPERATIONAL_MODE}

@app.get("/database/stats")
async def get_db_stats_endpoint():
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return get_db_stats()

@app.post("/agent/{agent_id}/calibrate")
async def calibrate_agent(agent_id: str, vk_ref: KatharosVector, db: Session = Depends(get_db)):
    # Legacy registry update
    agent_registry[agent_id] = get_dmr_instance(agent_id)

    # DB Persistence
    if OPERATIONAL_MODE in ["FULL", "DATABASE ONLY"]:
        repo = DMRRepository(db)
        if not repo.agent_exists(agent_id):
            repo.create_agent(agent_id, vk_ref.dict())
            return {"status": "calibrated", "agent_id": agent_id, "persistence": "saved"}
        return {"status": "already_calibrated", "agent_id": agent_id, "persistence": "verified"}

    return {"status": "calibrated", "agent_id": agent_id, "persistence": "none"}

@app.get("/agent/{agent_id}/vk", response_model=List[StateLayer])
async def get_vk_trajectory(agent_id: str, db: Session = Depends(get_db)):
    if OPERATIONAL_MODE in ["FULL", "DATABASE ONLY"]:
        repo = DMRRepository(db)
        layers = repo.get_trajectory(agent_id)
        if layers:
            return [
                StateLayer(
                    timestamp=int(l.timestamp.timestamp()),
                    vk=KatharosVector(bio=l.bio, aff=l.aff, soc=l.soc, cog=l.cog),
                    delta_k=l.delta_k,
                    q=l.q_value,
                    intensity=0.5 # Default intensity
                ) for l in layers
            ]

    # Fallback to Memory/Mock
    if agent_id not in agent_registry:
        agent_registry[agent_id] = get_dmr_instance(agent_id)

    ring = agent_registry[agent_id]
    trajectory = await asyncio.to_thread(ring.reconstruct_trajectory)

    result = []
    for layer in trajectory:
        if isinstance(layer, dict):
            result.append(StateLayer(
                timestamp=layer["timestamp"],
                vk=KatharosVector(bio=layer["bio"], aff=layer["aff"], soc=layer["soc"], cog=layer["cog"]),
                delta_k=layer.get("delta_k", 0.0),
                q=layer.get("q", 0.95),
                intensity=0.5
            ))
        else:
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

@app.post("/physics/s2/run_analysis")
async def run_s2_analysis(file_pattern: str, threshold: float = 0.05, output: str = "candidates.parquet"):
    def _run():
        loader = LHCDataLoader(file_pattern)
        analysis = ArkheLHCAnalysis(loader, s2_trigger)
        return analysis.run(trigger_threshold=threshold, output_file=output)

    candidates = await asyncio.to_thread(_run)
    return {"status": "completed", "candidates_count": len(candidates), "output": output}

@app.get("/physics/quantum/decoherence")
async def get_orbital_decoherence(h: float = 400e3):
    return {"tau_c": quantum_sim.orbital_decoherence(h)}

@app.post("/blockchain/satoshi/verify")
async def verify_satoshi(blocks: List[Dict]):
    return await verify_satoshi_temporal(blocks)

@app.get("/quantum/qiskit/trefoil_knot")
async def get_trefoil_knot():
    circuit = trefoil_knot_circuit()
    counts = await asyncio.to_thread(qiskit_iface.run_simulation, circuit)

    # Análise de Auto-Consistência e Nucleação
    nucleation = detect_wave_cloud_nucleation(counts)

    total_shots = sum(counts.values())
    p_00 = counts.get('000000', 0) / total_shots
    p_11 = counts.get('000011', 0) / total_shots

    coherence = p_00 + p_11

    return {
        "status": "success",
        "counts": counts,
        "p_00": p_00,
        "p_11": p_11,
        "coherence": coherence,
        "loop_closed": coherence > 0.7,
        "wave_cloud": nucleation,
        "qasm": qasm2.dumps(circuit)
    }

@app.get("/quantum/qiskit/novikov_loop")
async def get_novikov_loop(xi: float, dt: float, n_qubits: int = 2, use_kraus: bool = False):
    if use_kraus:
        circuit = novikov_loop_kraus(xi, dt, n_qubits_main=n_qubits)
    else:
        circuit = novikov_loop_circuit(xi, dt, n_qubits)

    counts = await asyncio.to_thread(qiskit_iface.run_simulation, circuit)
    return {
        "params": {"xi": xi, "dt": dt, "n_qubits": n_qubits, "use_kraus": use_kraus},
        "counts": counts,
        "qasm": qasm2.dumps(circuit)
    }

@app.post("/quantum/qiskit/submit")
async def submit_quantum_job(xi: float, dt: float, token: str = None):
    circuit = novikov_loop_kraus(xi, dt)
    result = qiskit_iface.submit_job(circuit, token)
    return result

@app.post("/knowledge/scan")
async def scan_semantic_anomalies(data: Dict[str, List[float]], threshold: float = 1.5):
    import pandas as pd

    def _run():
        df = pd.DataFrame(data)
        miner = SemanticMiner(df)
        return miner.detect_anomalies(threshold=threshold)

    anomalies = await asyncio.to_thread(_run)
    return {"status": "success", "anomalies_detected": anomalies}

@app.get("/knowledge/concept/{concept}")
async def analyze_concept(concept: str, values: List[float]):
    import pandas as pd

    def _run():
        df = pd.DataFrame({concept: values})
        miner = SemanticMiner(df)
        return miner.analyze_knowledge_squeezing(concept)

    return await asyncio.to_thread(_run)

@app.get("/monitoring/reality")
async def get_reality_status():
    return reality_listener.get_state()

@app.get("/metrics/synchronicity")
async def get_synchronicity(db: Session = Depends(get_db)):
    """
    Calcula o Índice de Sincronicidade (S) baseado na Tese Arkhe(n).
    Fórmula: S = (1 / ΔK) * P_AC
    """
    phi_q_threshold = 4.64
    delta_k = system_state.delta_k
    q_value = system_state.q_value

    phi_q_actual = q_value * 5.0 # Proxy: Q=1.0 -> φ_q=5.0

    if delta_k <= 0.0001:
        s_index = 100.0
    else:
        s_index = (1.0 / delta_k) * (phi_q_actual / phi_q_threshold)

    if s_index > 8.0:
        status = "SINGULARITY_IMMINENT"
    elif s_index > 5.0:
        status = "DIALOGUE_ACTIVE"
    elif s_index > 2.0:
        status = "AWAKENING"
    else:
        status = "DORMANT"

    metrics = {
        "synchronicity_index": s_index,
        "delta_k": delta_k,
        "phi_q_actual": phi_q_actual,
        "miller_threshold": phi_q_threshold,
        "wave_cloud_nucleated": phi_q_actual > phi_q_threshold,
        "status": status,
        "lmt": {
            "resonance": random.uniform(0.1, 0.9),
            "phase": 1,
            "currents": {
                "source": 0.8,
                "vibration": 0.6,
                "resonance": 0.7
            }
        },
        "p_ac_proxy": q_value,
        "thresholds": {
            "awakening": 2.0,
            "dialogue": 5.0,
            "singularity": 8.0
        }
    }

    # Log to Database
    if OPERATIONAL_MODE in ["FULL", "DATABASE ONLY"]:
        try:
            repo = DMRRepository(db)
            sys_metrics = repo.compute_system_metrics()
            repo.log_synchronicity(
                s_index=s_index,
                status=status,
                delta_k_avg=sys_metrics["avg_delta_k"] or delta_k,
                p_ac_proxy=q_value,
                total_agents=sys_metrics["total_agents"],
                ghost_count=random.randint(0, 5), # Simulated
                lambda_sync=random.uniform(0.5, 0.95), # Simulated
                h_value=repo.compute_h_value()
            )
        except Exception as e:
            print(f"Sync logging failed: {e}")

    return metrics

# ArkheOS Lite Components (Mocked for Python Integration)
class MockScheduler:
    def __init__(self):
        self.tasks = []
        self.phi_q = 1.0

    def add_task(self, task: Dict):
        self.tasks.append(task)
        # Simulate phi_q growth
        self.phi_q += task.get("coherence", 0) * 0.1

    def get_status(self):
        return {"phi_q": self.phi_q, "queue_len": len(self.tasks)}

scheduler_instance = MockScheduler()
handover_ledger = []

@app.post("/arkhe/scheduler/task")
async def submit_task(task: Dict[str, Any]):
    scheduler_instance.add_task(task)
    return {"status": "scheduled", "phi_q": scheduler_instance.phi_q}

@app.get("/arkhe/scheduler/status")
async def get_scheduler_status():
    return scheduler_instance.get_status()

@app.post("/arkhe/ledger/handover")
async def record_handover(handover: Dict[str, Any]):
    handover_ledger.append(handover)
    return {"status": "recorded", "ledger_size": len(handover_ledger)}

@app.get("/arkhe/ledger/history")
async def get_ledger_history():
    return handover_ledger

@app.websocket("/ws/reality")
async def websocket_reality(websocket: WebSocket):
    await websocket.accept()
    try:
        try:
            import arkhe_core
            topology_engine = arkhe_core.topology.KleinBottlehole()
            TOPOLOGY_AVAILABLE = True
        except ImportError:
            TOPOLOGY_AVAILABLE = False

        iterations = 0
        while True:
            iterations += 1
            state = reality_listener.get_state()

            if TOPOLOGY_AVAILABLE:
                is_open = topology_engine.check_monodromy_iteration(iterations)
                state["topology_status"] = "KLEIN_OPEN" if is_open else "CLOSED"
                if is_open:
                    state["state"] = "SINGULARITY_KLEIN"
                    state["q_value"] = 0.99

            await websocket.send_json(state)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass

@app.post("/bio/telemetry")
async def receive_bio_telemetry(agent_id: str, x: float, y: float, db: Session = Depends(get_db)):
    """
    Receives bio-telemetry (e.g. cardiac variance X and system density Y).
    Updates Transfer Entropy estimator and Hydraulic Engine for the agent.
    """
    if agent_id not in te_estimators:
        if RUST_AVAILABLE:
            import dmr_bridge
            te_estimators[agent_id] = dmr_bridge.PyTransferEntropy(10, 1000)
            hydraulic_engines[agent_id] = dmr_bridge.PyHydraulicEngine()
        else:
            # Fallback for mock/memory modes
            pass

    te_val = 0.0
    pressure = 0.0
    status_label = "DORMANT"

    if RUST_AVAILABLE:
        te_estimator = te_estimators[agent_id]
        h_engine = hydraulic_engines[agent_id]

        await asyncio.to_thread(te_estimator.add_observation, x, y)
        te_val = await asyncio.to_thread(te_estimator.calculate_te)

        # Update Hydraulic Engine: y is phi_q proxy, 1.0 - te_val is coherence proxy
        await asyncio.to_thread(h_engine.update, y, 1.0 - min(1.0, te_val))
        report = await asyncio.to_thread(h_engine.get_report)
        pressure = report.pressure
        status_label = str(report.state).split('.')[-1]

        # Coherence injection on high TE
        if te_val > 0.5:
            if agent_id in agent_registry:
                ring = agent_registry[agent_id]
                await asyncio.to_thread(ring.grow_layer, 0.5, 0.5, 0.5, 0.5, 0.99)

            # Save to Database
            if OPERATIONAL_MODE in ["FULL", "DATABASE ONLY"]:
                repo = DMRRepository(db)
                repo.add_layer(agent_id, 0.5, 0.5, 0.5, 0.5, 0.99, 0.01, 3600.0)
    else:
        te_val = random.uniform(0.1, 0.4)
        pressure = random.uniform(10.0, 50.0)

    # Sync with reality listener
    reality_listener.te_coupling = te_val
    reality_listener.hydraulic_state = {
        "state": status_label,
        "pressure": pressure,
        "flow_rate": 0.1,
        "viscosity": 0.5
    }

    return {
        "status": "success",
        "te_value": te_val,
        "hydraulic_state": status_label,
        "pressure": pressure
    }

@app.post("/geoloc/verify")
async def verify_location(agent_id: str, lat: float, lon: float, measurements: List[Dict], db: Session = Depends(get_db)):
    """
    Verifies location using BFT-PoLoc.
    """
    result = await asyncio.to_thread(geoloc_verifier.verify, lat, lon, measurements)

    coherence = 0.95 if result["is_valid"] else 0.3
    vk_val = 0.5 if result["is_valid"] else 0.7

    # Memory update
    if agent_id in agent_registry:
        ring = agent_registry[agent_id]
        await asyncio.to_thread(ring.grow_layer, vk_val, vk_val, vk_val, vk_val, coherence)

    # DB update
    if OPERATIONAL_MODE in ["FULL", "DATABASE ONLY"]:
        repo = DMRRepository(db)
        repo.add_layer(agent_id, vk_val, vk_val, vk_val, vk_val, coherence, 1.0 - coherence, 3600.0)

    return result

# --- Networking & QPU Endpoints ---

@app.get("/net/status")
async def get_net_status():
    """Returns the status of the P2P Horizontal Antenna."""
    return {
        "status": "active",
        "port": 7000,
        "peers_connected": 3,
        "protocol": "Teknet/Ω.224",
        "last_sync": int(time.time()) - 15
    }

@app.post("/net/broadcast")
async def broadcast_message(message: Dict):
    """Broadcasts a message to the Teknet P2P network."""
    # Simulation of P2P broadcast
    await asyncio.sleep(0.05)
    return {"status": "broadcast_sent", "message_id": f"msg_{random.randint(1000, 9999)}"}

@app.get("/qpu/status")
async def get_qpu_status():
    """Returns the status of the Vertical Antenna (Quantum Hardware)."""
    return {
        "backend": "ibm_brisbane",
        "status": "online",
        "phi_q": 4.64,
        "readout_error": 0.012,
        "last_calibration": "2023-10-27T10:00:00Z"
    }

@app.post("/qpu/calibrate")
async def calibrate_qpu():
    """Triggers a recalibration of the physical vacuum via QPU."""
    await asyncio.sleep(1.0) # Simulation
    return {"status": "recalibrated", "new_phi_q": 4.642}

@app.websocket("/ws/entrainment")
async def websocket_entrainment(websocket: WebSocket):
    await websocket.accept()
    agent_id = f"agent_{int(time.time())}"
    agent_registry[agent_id] = get_dmr_instance(agent_id)

    try:
        while True:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
