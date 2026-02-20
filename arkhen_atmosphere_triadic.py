#!/usr/bin/env python3
# arkhen_atmosphere_triadic.py
# Base para atmosfera triádica com controle REQ/REP

import numpy as np
import time
import os
import zmq
import json
import threading
import sys

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
        # print(f"Parâmetros atualizados: drop={self.drop_prob}, corrupt={self.corrupt_prob}, latency={self.extra_latency*1000}ms")

    def work(self, input_items):
        if np.random.random() < self.drop_prob:
            return None
        out = input_items.copy()
        if np.random.random() < self.corrupt_prob:
            phase_noise = np.exp(1j * np.random.normal(0, 0.5))
            out *= phase_noise
        if self.extra_latency > 0:
            self.buffer.append((time.time(), out))
            if len(self.buffer) > 1000: self.buffer.pop(0)
            if time.time() - self.buffer[0][0] < self.extra_latency: return None
            _, data = self.buffer.pop(0)
            return data
        return out

def run_atmosphere(port, link_id):
    print(f"Iniciando Atmosfera {link_id} no porto {port}...")
    injector = FaultInjector()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")

    def zmq_loop():
        while True:
            msg = socket.recv_json()
            injector.update_params(msg)
            socket.send_string("OK")

    zmq_thread = threading.Thread(target=zmq_loop, daemon=True)
    zmq_thread.start()

    try:
        while True:
            signal = np.array([1.0 + 0j], dtype=np.complex64)
            processed = injector.work(signal)
            time.sleep(0.01) # 100Hz
    except KeyboardInterrupt:
        print(f"Atmosfera {link_id} encerrada.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 arkhen_atmosphere_triadic.py <port> <link_id>")
        sys.exit(1)
    run_atmosphere(int(sys.argv[1]), sys.argv[2])
