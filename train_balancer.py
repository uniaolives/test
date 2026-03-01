from simulator.arkhe_sharded_env import ShardedClusterEnv
import numpy as np

def run_sample():
    env = ShardedClusterEnv({'num_shards': 4})
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.2f}")
        if terminated: break

if __name__ == "__main__":
    run_sample()
