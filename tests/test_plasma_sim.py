"""
Simulation Test for Plasma Cosmology model
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl_compiler import System, Node, Handover

def create_plasma_sim():
    sys = System("Plasma Cosmos Simulation")

    # Create Birkeland filaments
    f1 = Node("FilamentoBirkeland", corrente=10.0, raio=1.0, B_axial=np.array([0.0, 0.0, 1.0]), helicidade=0.5, coerência=0.9, potencial=5.0)
    f2 = Node("FilamentoBirkeland", corrente=10.0, raio=1.0, B_axial=np.array([0.0, 0.0, 1.1]), helicidade=0.5, coerência=0.8, potencial=15.0)

    sys.add_node(f1)
    sys.add_node(f2)

    # Handover: Double Layer (DuplaCamada)
    # condition: salto_de_potencial(a, b) > limiar;
    limiar = 8.0

    dl = Handover("DuplaCamada", "FilamentoBirkeland", "FilamentoBirkeland")
    def dl_cond(a, b):
        return abs(a.potencial - b.potencial) > limiar

    def dl_effect(a, b):
        print(f"  [HANDOVER] Double Layer active between {a.id} and {b.id}!")
        a.coerência += 0.01
        b.coerência += 0.01
        # Energy transfer
        diff = b.potencial - a.potencial
        a.potencial += diff * 0.1
        b.potencial -= diff * 0.1

    dl.set_condition(dl_cond)
    dl.set_effects(dl_effect)
    sys.add_handover(dl)

    return sys

def run_plasma_sim():
    print("--- PLASMA COSMOLOGY SIMULATION ---")
    sim = create_plasma_sim()

    f1, f2 = sim.nodes

    for t in range(10):
        print(f"t={sim.time:02d} | F1 Pot: {f1.potencial:.2f}, Coer: {f1.coerência:.3f} | F2 Pot: {f2.potencial:.2f}, Coer: {f2.coerência:.3f}")
        sim.step()

    print("\nSimulation completed successfully.")

if __name__ == "__main__":
    run_plasma_sim()
