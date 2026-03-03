import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from papercoder_kernel.oloid.ritual import ActivationRitualV2

async def main():
    ritual = ActivationRitualV2()
    success = await ritual.execute()
    assert success is True
    print("ActivationRitualV2 verification PASSED")

if __name__ == "__main__":
    asyncio.run(main())
