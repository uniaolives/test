#!/usr/bin/env python3
"""
Arkhe(N) Chaos Orchestrator
Orquestra ataques complexos (Solar Flare + BGP Hijack) para testar o SafeCore e o AKF.
"""

import time
import random
import math
import zmq
import json

def run_chaos_scenario():
    context = zmq.Context()
    # No cenário real, isso enviaria comandos para o Flowgraph GNU Radio e para o Simulador Diplomático
    # Aqui, simularemos enviando pacotes stress para o simulador
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    print("🎭 [ORCHESTRATOR] Iniciando Orquestração do Caos...")

    # 1. Baseline
    print("\n📍 Cenário 1: Baseline. Céu limpo.")
    for i in range(5):
        send_handshake(socket, "baseline-node", phase=0.1, coherence=0.98)
        time.sleep(1)

    # 2. Solar Flare (Simulação de Ruído Branco / Baixa Coerência)
    print("\n☀️ Cenário 2: Solar Flare (Tempestade Solar).")
    print("Injetando ruído AWGN massivo. C_local caindo...")
    for i in range(10):
        # Baixa coerência mas fase nominal ainda presente sob o ruído
        # O AKF deve segurar a predição
        send_handshake(socket, "solar-storm-node", phase=0.2, coherence=0.3)
        time.sleep(0.5)

    # 3. BGP Hijack (Jitter Extremo / Vórtice Topológico)
    print("\n📡 Cenário 3: BGP Hijack & Jitter Extremo.")
    print("Fases saltando violentamente. Induzindo Vórtices...")
    for i in range(10):
        # Coerência oscilando e fase totalmente errática
        phase_chaos = random.uniform(-math.pi, math.pi)
        send_handshake(socket, "hijack-node", phase=phase_chaos, coherence=random.uniform(0.1, 0.5))
        time.sleep(0.5)

    # 4. Recovery (Annealing)
    print("\n🛡️ Cenário 4: Recuperação. Escudos levantados.")
    print("Cessando interferência. Iniciando Annealing...")
    for i in range(15):
        send_handshake(socket, "recovery-node", phase=0.1, coherence=0.95)
        time.sleep(1)

    print("\n🏁 [ORCHESTRATOR] Orquestração finalizada.")

def send_handshake(socket, node_id, phase, coherence):
    request = {
        "type": "HANDSHAKE_REQUEST",
        "node_id": node_id,
        "phase_remote": phase,
        "coherence_local": coherence,
        "remote_coherence_sim": coherence # Simplificação
    }
    socket.send_json(request)
    response = socket.recv_json()
    print(f"[{node_id}] State: {response.get('protocol_state')} | α: {response.get('alpha'):.3f} | C: {response.get('coherence_global'):.3f}")

if __name__ == '__main__':
    try:
        run_chaos_scenario()
    except Exception as e:
        print(f"❌ Erro no orquestrador: {e}")
        print("Certifique-se de que o diplomatic_simulator.py está rodando.")
