#!/usr/bin/env python3
# arkhen_atmosphere.py
# Simulação de atmosfera com injeção de falhas para teste HWIL e controle ZMQ

import numpy as np
import time
import os
import zmq
import json
import threading

class FaultInjector:
    def __init__(self, drop_prob=0.0, corrupt_prob=0.0, extra_latency_ms=0):
        self.drop_prob = drop_prob
        self.corrupt_prob = corrupt_prob
        self.extra_latency = extra_latency_ms / 1000.0  # segundos
        self.buffer = []
        self.last_time = time.time()

    def update_params(self, params):
        self.drop_prob = params.get('drop_prob', self.drop_prob)
        self.corrupt_prob = params.get('corrupt_prob', self.corrupt_prob)
        self.extra_latency = params.get('extra_latency_ms', self.extra_latency * 1000.0) / 1000.0
        print(f"Parâmetros atualizados: drop={self.drop_prob}, corrupt={self.corrupt_prob}, latency={self.extra_latency*1000}ms")

    def work(self, input_items):
        # Aplica perda e corrupção
        if np.random.random() < self.drop_prob:
            return None # descarta

        out = input_items.copy()

        if np.random.random() < self.corrupt_prob:
            # Adiciona ruído de fase (erro de timestamp simulado)
            phase_noise = np.exp(1j * np.random.normal(0, 0.5))
            out *= phase_noise

        # Simula latência extra atrasando a entrega
        if self.extra_latency > 0:
            self.buffer.append((time.time(), out))
            # Remover itens muito antigos do buffer para evitar vazamento
            if len(self.buffer) > 1000:
                self.buffer.pop(0)

            # Se o primeiro item não está pronto, espera ou retorna vazio
            if time.time() - self.buffer[0][0] < self.extra_latency:
                return None

            _, data = self.buffer.pop(0)
            return data

        return out

def zmq_control_loop(injector):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print("ZMQ Control Loop iniciado (aguardando comandos em localhost:5556)...")

    while True:
        try:
            message = socket.recv_json()
            injector.update_params(message)
        except Exception as e:
            print(f"Erro no ZMQ: {e}")
            time.sleep(1)

def run_atmosphere():
    print("Iniciando Simulação de Atmosfera Arkhe(N)...")
    injector = FaultInjector(drop_prob=0.01, extra_latency_ms=20)

    # Iniciar thread de controle ZMQ
    control_thread = threading.Thread(target=zmq_control_loop, args=(injector,), daemon=True)
    control_thread.start()

    try:
        while True:
            # Simular geração de sinal
            signal = np.array([1.0 + 0j], dtype=np.complex64)
            processed = injector.work(signal)
            if processed is not None:
                # Simular envio para o transdutor
                pass
            time.sleep(0.01) # 100Hz
    except KeyboardInterrupt:
        print("Atmosfera encerrada.")

if __name__ == "__main__":
    run_atmosphere()
