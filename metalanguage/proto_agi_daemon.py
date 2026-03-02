#!/usr/bin/env python3
# proto_agi_daemon.py
# Daemon para Proto‑AGI baseado em ANL

import asyncio
import threading
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import sys
import os

# Add current directory to path to ensure anl is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from anl import *  # importa todo o módulo ANL

# ======================================================
# Configuração do Hipergrafo
# ======================================================

# Criar o hipergrafo central
hg = Hypergraph("ProtoAGI")

# Criar espaços de estado
scalar_space = StateSpace(1, "euclidean", "real")
embed_space = StateSpace(384, "euclidean", "real")

# Criar agentes (inicialmente sem LLM real, apenas placeholders)
alice = Node("Alice", scalar_space, 0)
bob = Node("Bob", scalar_space, 0)

# Memória partilhada (vectorial)
memory = SharedMemory("SharedMemory", storage_type="vector", dim=384)

# Adicionar ao hipergrafo
hg.add_node(alice).add_node(bob).add_node(memory)

# Handover: Alice incrementa o contador de Bob (simples comunicação)
handover_ab = Handover("AliceToBob", alice, bob, protocol=Protocol.CREATIVE)
handover_ab.set_mapping(lambda state: state + 1)  # Bob recebe state = state + 1
hg.add_handover(handover_ab)

# Handover: Bob escreve na memória partilhada
def bob_write_to_memory(handover, ctx):
    memory.write(f"bob_{time.time()}", bob.state, np.random.randn(384))
handover_bm = Handover("BobToMemory", bob, memory, protocol=Protocol.CREATIVE)
handover_bm.add_effect(bob_write_to_memory)
hg.add_handover(handover_bm)

# Handover: agente pode ler da memória (apenas efeito, sem mapeamento)
def read_from_memory(handover, ctx):
    # Simula uma consulta
    results = memory.query_vector(np.random.randn(384), top_k=3)
    # print(f"Memory query results: {results}")
handover_mb = Handover("MemoryToBob", memory, bob, protocol=Protocol.CONSERVATIVE)
handover_mb.add_effect(read_from_memory)
hg.add_handover(handover_mb)

# ======================================================
# Loop principal (executa em thread separada)
# ======================================================

class DaemonLoop:
    def __init__(self, hypergraph, dt=0.1):
        self.hg = hypergraph
        self.dt = dt
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.phi_history = []

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _run(self):
        step = 0
        while self.running:
            # Evoluir sistema
            self.hg.evolve(self.dt, steps=1)
            # Calcular φ (integração de informação)
            phi = self.hg.compute_integration()
            self.phi_history.append((time.time(), phi))
            step += 1
            time.sleep(self.dt)

    def get_status(self):
        return {
            "nodes": len(self.hg.nodes),
            "handovers": len(self.hg.handovers),
            "global_coherence": float(self.hg.global_coherence),
            "last_phi": float(self.phi_history[-1][1]) if self.phi_history else 0.0
        }

# Iniciar loop
daemon = DaemonLoop(hg, dt=0.1)
daemon.start()

# ======================================================
# API REST (FastAPI)
# ======================================================

app = FastAPI(title="Proto-AGI Daemon API")

class AgentStateResponse(BaseModel):
    agent_id: str
    state: float
    coherence: float

class HandoverRequest(BaseModel):
    handover_id: str
    context: Optional[Dict[str, Any]] = None

class MemoryWriteRequest(BaseModel):
    key: str
    value: Any
    embedding: Optional[List[float]] = None

class MemoryQueryRequest(BaseModel):
    embedding: List[float]
    top_k: int = 5

@app.get("/status", response_model=Dict)
async def get_status():
    return daemon.get_status()

@app.get("/agents", response_model=List[AgentStateResponse])
async def list_agents():
    states = []
    for node_id, node in hg.nodes.items():
        if isinstance(node.state, (int, float, np.number)):
            state_val = float(node.state)
        else:
            state_val = 0.0
        states.append(AgentStateResponse(
            agent_id=node_id,
            state=state_val,
            coherence=node.local_coherence
        ))
    return states

@app.post("/handover/{handover_id}")
async def trigger_handover(handover_id: str, request: Optional[HandoverRequest] = None):
    if handover_id not in hg.handovers:
        raise HTTPException(404, "Handover not found")
    handover = hg.handovers[handover_id]
    ctx = request.context if request else {}
    result = handover.execute(context=ctx)
    return {"status": "executed", "result": str(result)}

@app.post("/memory/write")
async def memory_write(req: MemoryWriteRequest):
    emb = np.array(req.embedding) if req.embedding else np.random.randn(384)
    memory.write(req.key, req.value, emb)
    return {"status": "written", "key": req.key}

@app.post("/memory/query")
async def memory_query(req: MemoryQueryRequest):
    emb = np.array(req.embedding)
    results = memory.query_vector(emb, req.top_k)
    # results is List[Tuple[str, Any]], converting for JSON
    serializable_results = [{"key": r[0], "value": str(r[1])} for r in results]
    return {"results": serializable_results}

@app.get("/phi")
async def get_phi():
    return {"phi_history": daemon.phi_history}

@app.on_event("shutdown")
def shutdown_event():
    daemon.stop()

if __name__ == "__main__":
    # Use loop=none to avoid issues if running in an existing loop,
    # but here we just run it.
    uvicorn.run(app, host="0.0.0.0", port=8000)
