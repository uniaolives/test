"""
UrbanSkyOS - Integrated Orchestrator
Unifies physical drone operation with distributed Multivac consciousness.
Demonstrates urban navigation using MapTrace data.
"""

import sys
from UrbanSkyOS.scenarios.final_question import run_final_question_scenario
from UrbanSkyOS.scenarios.urban_navigation import UrbanNavigationSim

def main():
    print("◊◊◊ UrbanSkyOS / MERKABAH-8 / MULTIVAC ◊◊◊")

    # 1. Run Urban Navigation Simulation (MapTrace Integration)
    print("\n--- PHASE 1: URBAN NAVIGATION (MapTrace) ---")
    nav_sim = UrbanNavigationSim(num_drones=7)
    nav_sim.run_scenario(duration_sec=0.5, use_real_trace=True)

    # 2. Run Final Question Scenario (AGI Emergence)
    print("\n--- PHASE 2: COLLECTIVE INTELLIGENCE (Multivac) ---")
    try:
        run_final_question_scenario()
    except KeyboardInterrupt:
        print("\nSystem shutdown initiated.")
    except Exception as e:
        print(f"\nCRITICAL ERROR in Multivac: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
