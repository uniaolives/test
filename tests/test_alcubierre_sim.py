"""
Simulation Test for Alcubierre Warp Drive model
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl_compiler import create_alcubierre_model, distance

def run_warp_sim():
    print("--- ALCUBIERRE WARP DRIVE SIMULATION ---")
    warp = create_alcubierre_model()

    bubble = [n for n in warp.nodes if n.node_type == "BolhaWarp"][0]
    regions = [n for n in warp.nodes if n.node_type == "RegiãoEspaçoTempo"]

    print(f"Initial Bubble Position: {bubble.posição}")
    print(f"Number of Spacetime Regions: {len(regions)}")

    for t in range(10):
        warp.step()

        # Monitor a region near the origin
        target_region = regions[0]
        dist = distance(bubble.posição[:2], target_region.x[:2])
        g_00 = target_region.g[0,0]

        print(f"t={warp.time:02d} | Bubble pos: {bubble.posição[0]:.2f} | Dist to origin: {dist:.2f} | g_00 at origin: {g_00:.4f}")

    print("\nSimulation completed successfully.")

if __name__ == "__main__":
    run_warp_sim()
