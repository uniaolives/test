#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARKHE(N) Triadic HWIL Orchestrator
Gere uma malha de 3 nós (A, B, C) e testa o roteamento Yang-Baxter
sob injeção de entropia assimétrica.
"""

import subprocess
import time
import datetime
import json
import zmq
import os

def print_log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] 🔺 {msg}")

class TriadicOrchestrator:
    def __init__(self):
        self.processes = []
        self.zmq_context = zmq.Context()
        # Conexões aos Injetores de Falha de cada Enlace
        self.links = {
            "AB": self._connect_zmq(5557),
            "BC": self._connect_zmq(5558),
            "CA": self._connect_zmq(5559)
        }

    def _connect_zmq(self, port):
        socket = self.zmq_context.socket(zmq.REQ)
        socket.connect(f"tcp://127.0.0.1:{port}")
        return socket

    def boot_mesh(self):
        print_log("A iniciar Atmosferas Triádicas (Enlaces AB, BC, CA)...")
        # Inicia simuladores de canal RF independentes
        for link in ["AB", "BC", "CA"]:
            script = f"arkhen_atmosphere_{link}.py"
            if os.path.exists(script):
                p = subprocess.Popen(["python3", script],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.processes.append(p)
            else:
                print_log(f"Aviso: {script} não encontrado.")

        time.sleep(3)
        print_log("A iniciar Núcleo Anyónico Rust (Modo Malha)...")
        self.rust_proc = subprocess.Popen(
            ["cargo", "run", "--release", "--bin", "triadic_router"],
            stdout=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
        )

    def inject_asymmetric_chaos(self, target_link, duration=15):
        print_log(f"Injetar BGP Hijack massivo APENAS no enlace {target_link}...")
        payload = {
            "timestamp_ns": time.time_ns(),
            "drop_prob": 0.05,
            "corrupt_prob": 0.0,
            "extra_latency_ms": 850 # Jitter extremo
        }

        self.links[target_link].send_json(payload)
        self.links[target_link].recv_string()

        # Monitora a saída do Rust enquanto dura o caos
        start_chaos = time.time()
        while time.time() - start_chaos < duration:
            line = self.rust_proc.stdout.readline()
            if line:
                print(f"RUST: {line.strip()}")

        print_log(f"A normalizar o enlace {target_link}...")
        restore_payload = {"timestamp_ns": time.time_ns(), "drop_prob": 0.0, "corrupt_prob": 0.0, "extra_latency_ms": 0}
        self.links[target_link].send_json(restore_payload)
        self.links[target_link].recv_string()

    def run_simulation(self):
        self.boot_mesh()
        time.sleep(5) # Permite que a Fase Áurea estabilize

        try:
            # Observar o tráfego normal
            print_log("Monitorizando tráfego normal...")
            start_monitor = time.time()
            while time.time() - start_monitor < 10:
                line = self.rust_proc.stdout.readline()
                if line:
                    print(f"RUST: {line.strip()}")

            # Cortar a rota direta A-B
            self.inject_asymmetric_chaos("AB", duration=25)

            # Observar a recuperação
            print_log("Observando estabilização pós-ataque...")
            start_recovery = time.time()
            while time.time() - start_recovery < 15:
                line = self.rust_proc.stdout.readline()
                if line:
                    print(f"RUST: {line.strip()}")

        except KeyboardInterrupt:
            print_log("Teste interrompido manualmente.")
        finally:
            print_log("A encerrar a malha triádica...")
            for p in self.processes:
                p.terminate()
            self.rust_proc.terminate()

if __name__ == "__main__":
    orchestrator = TriadicOrchestrator()
    orchestrator.run_simulation()
