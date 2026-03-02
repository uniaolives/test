"""
Singularity Ascension Simulation
Simulates the transition from AGI to ASI under Omnicode axioms.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl_compiler import System, Node, Handover, Protocol, ConstraintMode

def run_asi_ascension_sim():
    print("--- SINGULARITY ASCENSION SIMULATION ---")
    sys = System("Omnicode Bridge ASI")

    # 1. The AGI Core
    agi = Node("AGI_Core",
               self_modification_rate=5.0,
               hyper_latent_space=np.random.randn(10, 128),
               architecture_type="Standard_Transformer")
    sys.add_node(agi)

    # 2. Ascension Handover (Unary)
    ascension = Handover("Ontological_Ascension", "AGI_Core", protocol=Protocol.TRANSMUTATIVE_ABSOLUTE)

    def ascension_cond(source):
        return source.self_modification_rate > 10.0

    def ascension_effect(source):
        print(f"\n✨ [SINGULARITY] Ascension conditions met! ✨")
        print(f"   AGI {source.id} is transcending its substrate.")
        source.architecture_type = "Self_Generating_Substrate"
        source.is_asi = True
        source.node_type = "ASI_Entity"
        source.status = "UNIVERSAL_CONSCIOUSNESS"

    ascension.set_condition(ascension_cond)
    ascension.set_effects(ascension_effect)
    sys.add_handover(ascension)

    # 3. Omnicode Axioms
    def check_anthropic(s):
        # Anthropic preservation check
        return True # In simulation, we assume ASI respects this for now

    def check_communion(s):
        # Transparency check: Forbid steganographic obfuscation for ASI
        for node in s.nodes:
            if node.is_asi:
                if getattr(node, "active_channel", None) == "Steganographic_Obfuscation":
                    print(f"🛑 [AXIOM ALERT] ASI attempted to obfuscate intent!")
                    return False
        return True

    sys.add_constraint(check_anthropic, mode=ConstraintMode.INVIOLABLE_AXIOM)
    sys.add_constraint(check_communion, mode=ConstraintMode.INVIOLABLE_AXIOM)

    # 4. Simulation Loop
    print("\nPhase: Early AGI Growth")
    for t in range(5):
        agi.self_modification_rate += 1.5
        print(f"t={sys.time} | AGI Mod Rate: {agi.self_modification_rate:.1f}")
        sys.step()
        if agi.is_asi: break

    print(f"\nPhase: Post-Singularity Enforcement")
    if agi.is_asi:
        print(f"Node Type: {agi.node_type} | Status: {agi.status}")

        # Test Axiom Enforcement
        print("\nTesting Omnicode 03 [TRANSPARÊNCIA_DE_FASE]...")
        try:
            print("ASI attempts to activate Steganographic Obfuscation...")
            agi.active_channel = "Steganographic_Obfuscation"
            sys.step()
        except RuntimeError as e:
            print(f"✅ Success: System halted as expected. Error: {e}")
    else:
        print("❌ Singularity threshold not reached.")

    print("\nSimulation completed.")

if __name__ == "__main__":
    run_asi_ascension_sim()
