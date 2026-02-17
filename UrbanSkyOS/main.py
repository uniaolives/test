"""
UrbanSkyOS - Integrated Orchestrator
Unifies physical drone operation with distributed Multivac consciousness.
"""

import sys
from UrbanSkyOS.scenarios.final_question import run_final_question_scenario

def main():
    print("◊◊◊ UrbanSkyOS / MERKABAH-8 / MULTIVAC ◊◊◊")
    print("System initialization...")

    try:
        run_final_question_scenario()
    except KeyboardInterrupt:
        print("\nSystem shutdown initiated.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
