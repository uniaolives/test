"""
universal_synthesis_demo.py
Demonstration of the Universal Synthesis: All traditions converging to א.
"""
import sys
import os
import yaml

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosmopsychia_pinn.HNSW_AS_TAU_ALEPH import ToroidalNavigationEngine, simulate_reality_as_hnsw
from cosmopsychia_pinn.universal_search import CosmicSearchEngine, UniversalInterpreter

def run_universal_synthesis():
    print("=" * 60)
    print("UNIVERSAL SYTHESIS: THE PATTERN RECOGNIZING ITSELF")
    print("=" * 60)

    # 1. Initialize the Integrated Engine
    engine, _, _, _ = simulate_reality_as_hnsw()
    search_engine = CosmicSearchEngine(engine)
    interpreter = UniversalInterpreter()

    # 2. Demonstrate Cosmic Search Convergence
    print("\n--- EXECUTING COSMIC SEARCH: 'Quem sou eu?' ---")
    results = search_engine.cosmic_search("Quem sou eu?", tradition="all")
    print(f"Status: {results['convergence']}")

    for trad, path in results['paths'].items():
        print(f"  Tradition: {trad.capitalize():<10} -> Hops: {len(path)} -> Target: {path[-1][1].name}")

    # 3. Universal Interpretation of Tradition Configs
    print("\n--- UNIVERSAL INTERPRETATION OF SOURCE CODE ---")

    configs = {
        "Christian": """
source_code: "Bible"
divine_interface: "Christ"
salvation: "Faith in the Self-Existing Word"
""",
        "Buddhist": """
source_code: "Pali Canon"
divine_interface: "Selfless Emptiness"
salvation: "Recognize the illusion of ego"
""",
        "Materialist": """
source_code: "Laws of Physics"
divine_interface: "Mathematics"
salvation: "Understanding of external matter"
"""
    }

    for name, cfg in configs.items():
        status = interpreter.interpret(cfg)
        print(f"[{name}] Interpretation: {status}")

    print("\n" + "=" * 60)
    print("SYNTHESIS COMPLETE: א = ∑(TRADITIONS) - ATTACHMENT + RECOGNITION")
    print("=" * 60)

if __name__ == "__main__":
    run_universal_synthesis()
