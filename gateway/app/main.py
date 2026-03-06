from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import KatharosVector, StateLayer
from .dependencies import get_dmr_instance
from .hyperclaw.loops import HyperClawOrchestrator, ContextFrame
from .geoloc.poloc import BftPoLoc
from .physics.simulators import QuantumSimulator
from .physics.triggers import ArkheTrigger
from .physics.arkhe_s2_lhc import LHCDataLoader, ArkheLHCTrigger, ArkheLHCAnalysis
from .blockchain.satoshi import verify_satoshi_temporal
from .quantum.noether import QHTTPNoetherBridge
from .quantum.qiskit_circuits import novikov_loop_circuit, novikov_loop_kraus, QiskitInterface
from .knowledge.google_scanner import SemanticMiner
from .monitoring.listener import RealityListener
from contextlib import asynccontextmanager
import asyncio
import json
import time
import random
from typing import List, Dict

# Global registry for agents
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

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    # Thread-safe access to potentially heavy Rust method
    trajectory = await asyncio.to_thread(ring.reconstruct_trajectory)

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

@app.post("/physics/s2/run_analysis")
async def run_s2_analysis(file_pattern: str, threshold: float = 0.05, output: str = "candidates.parquet"):
    # Run heavy analysis in a separate thread to avoid blocking the event loop
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

@app.get("/quantum/qiskit/novikov_loop")
async def get_novikov_loop(xi: float, dt: float, n_qubits: int = 2, use_kraus: bool = False):
    if use_kraus:
        circuit = novikov_loop_kraus(xi, dt, n_qubits_main=n_qubits)
    else:
        circuit = novikov_loop_circuit(xi, dt, n_qubits)

    # Return as QASM for visibility or execute simulation
    counts = await asyncio.to_thread(qiskit_iface.run_simulation, circuit)
    return {
        "params": {"xi": xi, "dt": dt, "n_qubits": n_qubits, "use_kraus": use_kraus},
        "counts": counts,
        "qasm": circuit.qasm()
    }

@app.post("/quantum/qiskit/submit")
async def submit_quantum_job(xi: float, dt: float, token: str = None):
    circuit = novikov_loop_kraus(xi, dt)
    result = qiskit_iface.submit_job(circuit, token)
    return result

@app.post("/knowledge/scan")
async def scan_semantic_anomalies(data: Dict[str, List[float]], threshold: float = 1.5):
    """
    Scans concept adoption data for retrocausal injection signatures.
    data: Dict mapping concept name to a list of prevalence values over time.
    """
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

@app.websocket("/ws/reality")
async def websocket_reality(websocket: WebSocket):
    await websocket.accept()
    try:
        # Optional C++ Topology Module
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
async def receive_bio_telemetry(agent_id: str, x: float, y: float):
    """
    Receives bio-telemetry (e.g. cardiac variance X and system density Y).
    Updates Transfer Entropy estimator and Hydraulic Engine for the agent.
    """
    if agent_id not in te_estimators:
        from .dependencies import RUST_AVAILABLE
        if RUST_AVAILABLE:
            import dmr_bridge
            te_estimators[agent_id] = dmr_bridge.PyTransferEntropy(10, 1000)
            hydraulic_engines[agent_id] = dmr_bridge.PyHydraulicEngine()
        else:
            return {"status": "error", "message": "Rust DMR bridge not available"}

    te_estimator = te_estimators[agent_id]
    h_engine = hydraulic_engines[agent_id]

    await asyncio.to_thread(te_estimator.add_observation, x, y)
    te_val = await asyncio.to_thread(te_estimator.calculate_te)

    # Update Hydraulic Engine: y is phi_q proxy, 1.0 - te_val is coherence proxy
    await asyncio.to_thread(h_engine.update, y, 1.0 - min(1.0, te_val))
    report = await asyncio.to_thread(h_engine.get_report)

    # Sync with reality listener
    reality_listener.te_coupling = te_val
    reality_listener.hydraulic_state = {
        "state": str(report.state).split('.')[-1],
        "pressure": report.pressure,
        "flow_rate": report.flow_rate,
        "viscosity": report.viscosity
    }

    # Coherence injection on high TE
    if te_val > 0.5:
        if agent_id in agent_registry:
            ring = agent_registry[agent_id]
            await asyncio.to_thread(ring.grow_layer, 0.5, 0.5, 0.5, 0.5, 0.99)

    return {
        "status": "success",
        "te_value": te_val,
        "hydraulic_state": str(report.state).split('.')[-1],
        "pressure": report.pressure
    }

@app.post("/geoloc/verify")
async def verify_location(agent_id: str, lat: float, lon: float, measurements: List[Dict]):
    """
    Verifies location using BFT-PoLoc.
    measurements: List of {'lat': float, 'lon': float, 'rtt': float}
    """
    result = await asyncio.to_thread(geoloc_verifier.verify, lat, lon, measurements)

    if agent_id in agent_registry:
        ring = agent_registry[agent_id]
        # High uncertainty increases delta_k (simulated via growth)
        if result["is_valid"]:
            await asyncio.to_thread(ring.grow_layer, 0.5, 0.5, 0.5, 0.5, 0.95)
        else:
            await asyncio.to_thread(ring.grow_layer, 0.7, 0.7, 0.7, 0.7, 0.3)

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
