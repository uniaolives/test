#!/usr/bin/env python3
"""
6G Global Resilience Simulator: Shannon Fallback & Intercontinental Latency
"""

import numpy as np
import matplotlib.pyplot as plt

def run_global_sim():
    # Parameters
    distance_km = 9000 # Seoul <-> Grenoble
    c = 300000 # km/s
    propagation_delay = (distance_km / c) * 1000 # ~30ms one way

    battery = 1.0 # Joules
    critical_threshold = 0.05

    energy_semantic = 0.001 # J
    energy_classic = 0.012  # J (much higher)

    print(f"🌐 Intercontinental Link (Propagation Delay: {propagation_delay:.2f}ms)")

    results = []

    # Simulate increasing Context Drift (Ontological Noise)
    for epoch in range(60):
        drift = epoch / 60.0
        confidence = 1.0 - drift

        if confidence > 0.85:
            mode = "SEMANTIC"
            energy_step = energy_semantic
        else:
            mode = "SHANNON_FALLBACK"
            energy_step = energy_classic

        battery -= energy_step
        results.append({
            'epoch': epoch,
            'confidence': confidence,
            'mode': mode,
            'battery': battery
        })

        if battery < critical_threshold:
            print(f"Epoch {epoch}: 🛑 Battery Critical! Mode: {mode}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Mode={mode}, Confidence={confidence:.2f}, Battery={battery:.4f}")

    # Plotting
    epochs = [r['epoch'] for r in results]
    bats = [r['battery'] for r in results]
    confs = [r['confidence'] for r in results]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Battery (J)', color='tab:blue')
    ax1.plot(epochs, bats, color='tab:blue', label='Battery')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Confidence', color='tab:red')
    ax2.plot(epochs, confs, color='tab:red', linestyle='--', label='Confidence')
    ax2.axhline(y=0.85, color='gray', linestyle=':', label='Threshold')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title("6G Global Resilience: Semantic vs Shannon Fallback")
    fig.tight_layout()
    plt.savefig('6g_global_resilience_results.png')
    print("Results saved to 6g_global_resilience_results.png")

if __name__ == "__main__":
    run_global_sim()
