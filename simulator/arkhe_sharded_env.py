import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulator.blackout_generator import BlackoutGenerator

class ShardedClusterEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.num_shards = config.get('num_shards', 8)
        self.max_queue = config.get('max_queue', 100)

        self.action_space = spaces.Discrete(self.num_shards)
        obs_dim = self.num_shards * 3 + 2
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)

        self.blackout_gen = BlackoutGenerator(self.num_shards)
        self.shards = self._init_shards()
        self.step_count = 0

    def _init_shards(self):
        shards = []
        for i in range(self.num_shards):
            shards.append({
                'id': i,
                'queue_length': np.random.randint(0, 10),
                'latency': np.random.uniform(1, 5),
                'power': np.random.uniform(100, 200)
            })
        return shards

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shards = self._init_shards()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.blackout_gen.step()
        shard = self.shards[action]

        is_down = self.blackout_gen.is_blackout(action)
        if is_down:
            success = False
            latency = 100.0 # High latency for timeout
            energy_cost = 0.0
        else:
            success = np.random.random() > (shard['queue_length'] / self.max_queue)
            latency = shard['latency']
            energy_cost = shard['power'] * 0.001

        reward = self._compute_reward(success, latency, energy_cost, is_down)

        shard['queue_length'] = max(0, shard['queue_length'] + (1 if not success else -1))
        self.step_count += 1
        terminated = self.step_count >= 100

        return self._get_obs(), reward, terminated, False, {}

    def _compute_reward(self, success, latency, energy, is_down):
        reward = 1.0 if success else -1.0
        reward -= (latency / 10.0)
        reward -= energy * 0.1
        if is_down:
            reward -= 5.0 # Heavy penalty for choosing a dead shard
        return reward

    def _get_obs(self):
        obs = []
        for s in self.shards:
            obs.extend([s['queue_length']/self.max_queue, s['latency']/10.0, s['power']/500.0])
        obs.extend([0.5, 0.5]) # Global placeholders
        return np.array(obs, dtype=np.float32)
