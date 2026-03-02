import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
import os

# Add root to path to import anl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metalanguage.anl import PureCuriosityNode

# ======================================================
# AMBIENTE: GridWorld
# ======================================================

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = (0,0)
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)
        }
        # Fixed colors for each cell to be discovered
        self.colors = np.random.randint(1, 4, size=(size, size)) # Colors 1, 2, 3

    def step(self, action_str):
        dr, dc = self.action_effects[action_str]
        r, c = self.agent_pos
        nr = max(0, min(self.size-1, r + dr))
        nc = max(0, min(self.size-1, c + dc))
        self.agent_pos = (nr, nc)
        obs = self.colors[nr, nc]
        return obs

    def reset(self):
        self.agent_pos = (0,0)
        return self.colors[0,0]

# ======================================================
# SIMULAÇÃO
# ======================================================

def run_simulation(steps=100, size=5):
    env = GridWorld(size=size)
    n_states = size * size
    n_obs = 4 # 0, 1, 2, 3
    n_actions = 4

    agent = PureCuriosityNode("ScientistAgent", n_states, n_obs, n_actions)

    # Configure B matrix for the GridWorld
    B = np.zeros((n_states, n_states, n_actions))
    for s in range(n_states):
        r, c = divmod(s, size)
        for a_idx, action_str in enumerate(['up', 'down', 'left', 'right']):
            dr, dc = env.action_effects[action_str]
            nr, nc = max(0, min(size-1, r + dr)), max(0, min(size-1, c + dc))
            ns = nr * size + nc
            B[ns, s, a_idx] = 1.0
    agent.B = B

    obs = env.reset()
    trajectory = [env.agent_pos]
    entropies = []

    print(f"Starting Scientist Simulation on {size}x{size} Grid...")

    for i in range(steps):
        agent.step(obs)
        action_idx = agent.act()
        action_str = env.actions[action_idx]
        obs = env.step(action_str)

        trajectory.append(env.agent_pos)

        # Track total entropy of Matrix A (Global Uncertainty)
        A = agent.get_A_matrix()
        total_entropy = -np.sum(A * np.log(A + 1e-16))
        entropies.append(total_entropy)

        if i % 20 == 0:
            visited = len(set(trajectory))
            print(f"Step {i}: Pos={env.agent_pos}, Visited={visited}/{n_states}, Entropy={total_entropy:.2f}")

    print("Simulation Complete.")

    # Visualization
    traj = np.array(trajectory)
    plt.figure(figsize=(12, 5))

    # Plot 1: Trajectory
    plt.subplot(1, 2, 1)
    plt.imshow(env.colors, cmap='Pastel1', origin='upper')
    plt.plot(traj[:,1], traj[:,0], color='black', alpha=0.5, label='Path')
    plt.scatter(traj[:,1], traj[:,0], c=range(len(traj)), cmap='viridis', s=20)
    plt.scatter(0, 0, marker='*', color='red', s=100, label='Start')
    plt.title(f'Scientist Path ({len(set(trajectory))} cells discovered)')
    plt.legend()

    # Plot 2: Entropy Reduction
    plt.subplot(1, 2, 2)
    plt.plot(entropies)
    plt.title('Total Epistemic Uncertainty (Matrix A Entropy)')
    plt.xlabel('Steps')
    plt.ylabel('Entropy')

    plt.tight_layout()
    plt.savefig('pure_curiosity_results.png')
    print("Results saved to pure_curiosity_results.png")

if __name__ == "__main__":
    run_simulation(steps=150, size=5)
