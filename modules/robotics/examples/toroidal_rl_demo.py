# modules/robotics/examples/toroidal_rl_demo.py
import numpy as np
import sys
import os

# Add relevant paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../node/python')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../hal/python')))

from toroidal_policy import ToroidalPolicy
from eeg_toroidal_mapping import eeg_to_toroidal, human_swarm_coupling

def run_toroidal_rl_demo():
    print("--- Arkhe(n) Toroidal RL Demo ---")

    # Initialize policy with a constitutional winding target (1 poloidal cycle)
    policy = ToroidalPolicy(constitution_winding=(1, 0))

    # Human Operator Mock (Steady Focus)
    human_eeg = {'alpha': 0.4, 'beta': 1.5, 'theta': 0.2, 'delta': 0.05, 'gamma': 0.9}
    human_theta, human_phi = eeg_to_toroidal(**human_eeg)

    print(f"Human state locked at: {human_theta:.2f}, {human_phi:.2f}")

    # Simulation Loop
    dt = 0.1
    steps = 100
    for t in range(steps):
        # 1. Compute Reward Gradient (Simplistic: move towards high focus)
        # In a real case, this comes from task performance
        grad_theta = 0.5 * (human_theta - policy.theta)
        grad_phi = 0.5 * (human_phi - policy.phi)

        # 2. Update Policy (Coupled to human)
        policy.update([grad_theta, grad_phi], dt=dt)

        # 3. Output state every 20 steps
        if t % 20 == 0:
            speed, direction = policy.to_action()
            print(f"t={t:03d} | Policy: θ={policy.theta:.2f}, φ={policy.phi:.2f} | Winding: ({policy.w_poloidal}, {policy.w_toroidal}) | Speed: {speed:.2f}")

    print("\nDemo finished. Toroidal stability maintained.")

if __name__ == "__main__":
    run_toroidal_rl_demo()
