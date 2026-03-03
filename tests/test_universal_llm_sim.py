"""
Simulation Test for UniversalLLM model
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl_compiler import System, Node, Handover, sample

def create_llm_sim():
    sys = System("UniversalLLM Simulation")

    # Core
    core = Node("LLM_Core",
                weights=np.random.randn(10, 50), # 10 inputs -> 50 logits (vocab)
                architecture=lambda x, w: np.dot(x, w),
                alignment_score=0.95,
                capacity_entropy=0.7)

    # Context
    ctx = Node("ContextWindow",
               max_tokens=100,
               current_state=np.random.randn(128),
               attention_mask=np.eye(128))

    sys.add_node(core)
    sys.add_node(ctx)

    # Handover: ForwardPass
    fp = Handover("ForwardPass", "ContextWindow", "LLM_Core")
    def fp_cond(c, co):
        return len(c.current_state) > 0

    def fp_effect(c, co):
        print(f"  [INFERENCE] Forward Pass executing...")
        # Simulate logit generation and sampling
        logits = co.architecture(c.current_state[:10], co.weights)
        new_token_id = sample(logits, co.capacity_entropy)
        print(f"  [TOKEN] Generated token ID: {new_token_id}")
        # Append new "embedding" to state
        c.current_state = np.append(c.current_state, np.random.randn(1))

    fp.set_condition(fp_cond)
    fp.set_effects(fp_effect)
    sys.add_handover(fp)

    return sys

def run_llm_sim():
    print("--- UNIVERSAL LLM SIMULATION ---")
    sim = create_llm_sim()

    for t in range(5):
        print(f"t={sim.time:02d} | Context size: {len(sim.nodes[1].current_state)}")
        sim.step()

    print("\nSimulation completed successfully.")

if __name__ == "__main__":
    run_llm_sim()
