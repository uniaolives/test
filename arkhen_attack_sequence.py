#!/usr/bin/env python3
# arkhen_attack_sequence.py
# Injetor de ataques anyônicos via ZMQ

import zmq
import time
import json

def send_attack(socket, name, params):
    print(f"🔥 Lançando ataque: {name}...")
    socket.send_json(params)
    time.sleep(10) # Duração do ataque

def run_sequence():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")

    print("Injetor de Caos ZMQ pronto. Aguardando 5s para estabilização...")
    time.sleep(5)

    # 1. Tempestade Solar (Aumento de Ruído)
    send_attack(socket, "Tempestade Solar", {
        "drop_prob": 0.05,
        "corrupt_prob": 0.1,
        "extra_latency_ms": 50
    })

    # 2. Baseline (Recuperação)
    print("🌱 Cessando ataque. Aguardando recuperação...")
    socket.send_json({"drop_prob": 0.01, "corrupt_prob": 0.0, "extra_latency_ms": 10})
    time.sleep(15)

    # 3. Eclipse Orbital (Perda Massiva)
    send_attack(socket, "Eclipse Orbital", {
        "drop_prob": 0.3,
        "corrupt_prob": 0.05,
        "extra_latency_ms": 100
    })

    # 4. Baseline
    socket.send_json({"drop_prob": 0.01, "corrupt_prob": 0.0, "extra_latency_ms": 10})
    time.sleep(15)

    # 5. BGP Hijack (Instabilidade de Roteamento / Latência Extrema)
    send_attack(socket, "BGP Hijack", {
        "drop_prob": 0.1,
        "corrupt_prob": 0.2,
        "extra_latency_ms": 500
    })

    # Finalizar
    print("✅ Sequência de ataques concluída.")
    socket.send_json({"drop_prob": 0.01, "corrupt_prob": 0.0, "extra_latency_ms": 10})

if __name__ == "__main__":
    run_sequence()
