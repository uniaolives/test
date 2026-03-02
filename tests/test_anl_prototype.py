"""
Test and Demonstration of the ANL Prototype
"""

import sys
import os

# Add parent directory to path to find metalanguage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl_compiler import create_predator_prey

def run_demonstration():
    print("--- ARKHE(N) LANGUAGE (ANL) PROTOTYPE DEMO ---")
    print("Loading Predator-Prey Ecosystem model...")

    eco = create_predator_prey()
    print(f"Initial state: {eco}")
    for node in eco.nodes:
        print(f"  {node}")

    print("\nSimulating 20 time steps...")
    for t in range(20):
        eco.step()

        # Survival logic (usually part of dynamics but here we handle removal for the demo)
        dead = []
        for n in eco.nodes:
            if n.node_type in ["Coelho", "Raposa"] and n.energia <= 0:
                dead.append(n)
        for d in dead:
            eco.remove_node(d)

        coelhos = len([n for n in eco.nodes if n.node_type == "Coelho"])
        raposas = len([n for n in eco.nodes if n.node_type == "Raposa"])
        grama_bio = sum([n.biomassa for n in eco.nodes if n.node_type == "Grama"])

        print(f"t={eco.time:02d} | Coelhos: {coelhos} | Raposas: {raposas} | Grama: {grama_bio:.1f}")

    print("\nFinal state:")
    print(eco)
    print("Demo completed successfully.")

if __name__ == "__main__":
    run_demonstration()
