import numpy as np

class BlackoutGenerator:
    def __init__(self, num_shards, rate=0.01):
        self.num_shards = num_shards
        self.rate = rate
        self.active_blackouts = {}

    def step(self):
        if np.random.random() < self.rate:
            shard_id = np.random.randint(self.num_shards)
            duration = np.random.randint(10, 50)
            self.active_blackouts[shard_id] = duration
            print(f"⚡ BLACKOUT started on shard {shard_id} for {duration} steps")

        # Decay active blackouts
        for shard_id in list(self.active_blackouts.keys()):
            self.active_blackouts[shard_id] -= 1
            if self.active_blackouts[shard_id] <= 0:
                del self.active_blackouts[shard_id]

    def is_blackout(self, shard_id):
        return shard_id in self.active_blackouts
