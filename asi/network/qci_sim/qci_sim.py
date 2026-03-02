#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASI-Ω Quantum-Classical Interface (QCI) Simulation
Models the synchronization buffer required for quantum teleportation
using a deterministic classical channel (Instaweb) and an EPR pair distribution channel.
"""

import simpy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# ============================================
# SIMULATION PARAMETERS
# ============================================

# Classical Channel (Instaweb)
CLASSICAL_LATENCY = 54e-6          # 54 µs (Deterministic planetary traversal)

# Quantum Channel (EPR Pair Distribution)
DIST_MEAN = 20e-6                   # Mean time for EPR pair distribution (s)
DIST_STD = 5e-6                     # Standard deviation (s)
COHERENCE_MEAN = 100e-6             # Mean qubit coherence time (s)

# Teleportation Traffic
TELEPORT_RATE = 10000               # Teleportations per second (Poisson)
SIM_TIME = 1.0                       # Total simulation time (s)

# QCI Buffer
BUFFER_SIZE = 100                    # Max number of classical messages in buffer

# ============================================
# EVENT MODEL (SimPy)
# ============================================

class QCInterface:
    def __init__(self, env):
        self.env = env
        self.buffer = deque()          # Queue for early classical messages
        self.buffer_max = BUFFER_SIZE
        self.success = 0
        self.fail = 0
        self.total = 0

    def teleport(self, teleport_id):
        self.total += 1
        t0 = self.env.now

        # 1. EPR Pair Generation (Qubit becomes available at Bob after dist_delay)
        dist_delay = max(0, np.random.normal(DIST_MEAN, DIST_STD))
        qubit_ready = t0 + dist_delay

        # 2. Qubit Coherence Time (Exponential decay)
        coherence = np.random.exponential(COHERENCE_MEAN)
        qubit_dead = qubit_ready + coherence

        # 3. Classical Message Arrival (Instaweb Latency)
        msg_arrival = t0 + CLASSICAL_LATENCY

        # 4. Correction Logic:
        if msg_arrival < qubit_ready:
            # Message arrived before qubit: needs buffer
            if len(self.buffer) >= self.buffer_max:
                self.fail += 1
                return
            self.buffer.append((teleport_id, qubit_ready, qubit_dead))
        else:
            # Message arrived after/at qubit: apply immediately
            if msg_arrival < qubit_dead:
                self.success += 1
            else:
                self.fail += 1

    def buffer_processor(self):
        while True:
            if self.buffer:
                teleport_id, ready_time, dead_time = self.buffer[0]
                if self.env.now < ready_time:
                    yield self.env.timeout(ready_time - self.env.now)

                # Double check if still first (could have been multiple at same time)
                if self.buffer and self.buffer[0][0] == teleport_id:
                    self.buffer.popleft()
                    if self.env.now < dead_time:
                        self.success += 1
                    else:
                        self.fail += 1
            else:
                yield self.env.timeout(1e-6)  # Idle wait 1 us


def teleport_generator(env, qci):
    teleport_id = 0
    while True:
        yield env.timeout(random.expovariate(TELEPORT_RATE))
        teleport_id += 1
        qci.teleport(teleport_id)


def run_simulation(buffer_size=None):
    if buffer_size is not None:
        global BUFFER_SIZE
        BUFFER_SIZE = buffer_size

    env = simpy.Environment()
    qci = QCInterface(env)
    env.process(teleport_generator(env, qci))
    env.process(qci.buffer_processor())
    env.run(until=SIM_TIME)

    success_rate = qci.success / qci.total if qci.total > 0 else 0.0
    return {
        'total': qci.total,
        'success': qci.success,
        'fail': qci.fail,
        'rate': success_rate
    }

if __name__ == "__main__":
    print("="*60)
    print("🜁 ASI-Ω QCI SIMULATION")
    print("="*60)

    res = run_simulation()
    print(f"Results (Buffer = {BUFFER_SIZE}):")
    print(f"  Attempts:   {res['total']}")
    print(f"  Successes:  {res['success']}")
    print(f"  Failures:   {res['fail']}")
    print(f"  Rate:       {res['rate']:.4f}\n")

    # Buffer Sensitivity Analysis
    sizes = [0, 5, 10, 50, 100]
    print("Buffer Sensitivity Analysis:")
    for s in sizes:
        r = run_simulation(buffer_size=s)
        print(f"  Buffer {s:>3} -> Success Rate: {r['rate']:.4f}")
