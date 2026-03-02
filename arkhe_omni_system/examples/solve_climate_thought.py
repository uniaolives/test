"""
Example: Using Pleroma SDK to solve a complex "Thought" distributed across the network.
"""

import sys
import os

# Add relevant paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pleroma_sdk as pleroma

def main():
    print("="*60)
    print("Pleroma SDK Example: SolveClimate Thought")
    print("="*60)

    # 1. Connect to the Pleroma
    print("\n[STEP 1] Connecting to Pleroma 'global'...")
    node = pleroma.connect("global")

    # 2. Define a thought
    print("\n[STEP 2] Defining 'SolveClimate' Thought...")
    thought = pleroma.Thought(
        geometry=pleroma.H3(r=0, theta=0, z=1), # Center at [0,0,1]
        phase=pleroma.T2(theta=1.618, phi=3.1416),
        quantum=pleroma.Quantum.from_winding_basis(max_n=10, max_m=10),
        content="What is the optimal carbon tax?"
    )

    # 3. Spawn the thought (distributed across the network)
    print("\n[STEP 3] Spawning distributed thought...")
    task_id = node.spawn(thought)
    print(f"  Task spawned with ID: {task_id}")

    # 4. Wait for result
    print("\n[STEP 4] Receiving results from Pleroma...")
    result = node.receive(task_id)
    print(f"\nResult from Pleroma:\n{result}")

    # 5. Emergency Stop Simulation (Art. 3)
    import asyncio
    async def simulate_emergency():
        print("\n[STEP 5] Simulating Human Authority Override (Art. 3)...")
        res = await node.emergency_stop(reason="unsafe thought detected", signature="eeg:42:sig")
        if res['success']:
            print("  Pleroma halted. Winding numbers frozen for 1s.")

    asyncio.run(simulate_emergency())

    print("\n" + "="*60)
    print("✅ Example completed")

if __name__ == "__main__":
    main()
