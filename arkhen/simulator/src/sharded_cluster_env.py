import gymnasium as gym
from gymnasium import spaces
import numpy as np
from blackout_generator import BlackoutGenerator

class ShardedClusterEnv(gym.Env):
    def __init__(self, num_shards=10):
        super().__init__()
        self.num_shards = num_shards
        self.action_space = spaces.Discrete(num_shards)
        # Observation: [queue_lengths..., blackout_severities...]
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * num_shards,), dtype=np.float32)

        self.shards = [{"id": i, "queue": 0.0, "power": 0.5} for i in range(num_shards)]
        self.blackout_gen = BlackoutGenerator(num_shards)
        self.time = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        for s in self.shards: s["queue"] = 0.0
        self.blackout_gen.active = []
        return self._get_obs(), {}

    def _get_obs(self):
        queues = np.array([s["queue"] for s in self.shards], dtype=np.float32)
        severities = np.array([self.blackout_gen.is_affected(s["id"]) for s in self.shards], dtype=np.float32)
        return np.concatenate([queues, severities])

    def step(self, action):
        self.time += 1
        self.blackout_gen.step(self.time)

        chosen_shard = self.shards[action]
        severity = self.blackout_gen.is_affected(action)

        reward = 0.0
        if severity >= 1.0:
            reward = -10.0 # Catastrophic choice
            success = False
        else:
            success = np.random.random() > (chosen_shard["queue"] + severity * 0.5)
            if success:
                reward = 1.0 - (chosen_shard["power"] * (1.0 + severity))
                chosen_shard["queue"] = min(1.0, chosen_shard["queue"] + 0.1)
            else:
                reward = -1.0

        # Natural decay of queues
        for s in self.shards:
            s["queue"] = max(0.0, s["queue"] - 0.05)

        return self._get_obs(), reward, False, False, {"success": success}
