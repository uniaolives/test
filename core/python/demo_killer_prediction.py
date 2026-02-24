# demo_killer_prediction.py
import numpy as np
import matplotlib.pyplot as plt
from cds_framework.core.physics import PhiFieldSimulator
from cds_framework.plugins.pupillometry import predict_attentional_cost

def run_demo():
    print("Running Consciousness Dynamics Simulator (CDS) - Killer Prediction Demo")

    # Initialize simulator
    sim = PhiFieldSimulator(size=100, r=-1.0, u=1.0, gamma=1.0, dt=0.01)

    # Part 1: Baseline stability
    history_baseline = sim.simulate(steps=500, external_h=0.0)

    # Part 2: Parametric increase in attentional demand (external field H)
    history_demand = []
    h_values = [0.1, 0.5, 2.0]
    for h in h_values:
        history_h = sim.simulate(steps=200, external_h=h)
        history_demand.extend(history_h)

    full_history = np.concatenate([history_baseline, history_demand])

    # Predict attentional cost
    pupil_cost = predict_attentional_cost(full_history)

    # Analyze scaling: Cost vs (ΔΦ)²
    delta_phi = np.diff(full_history)
    squared_delta_phi = delta_phi**2

    print(f"Total steps simulated: {len(full_history)}")
    print(f"Average Order Parameter Φ: {np.mean(full_history):.4f}")

    # Save a verification plot (simulated)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(full_history, label='Order Parameter Φ')
    plt.title('Φ-field Dynamics')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(pupil_cost, color='orange', label='Predicted Pupil Dilation')
    plt.title('Predicted Attentional Cost (Pupillometry)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cds_killer_prediction.png')
    print("Demo complete. Result saved to cds_killer_prediction.png.")

if __name__ == "__main__":
    run_demo()
