import numpy as np

class BlackoutEvent:
    def __init__(self, shard_ids, start_time, duration, severity):
        self.shard_ids = shard_ids
        self.start_time = start_time
        self.duration = duration
        self.severity = severity

class BlackoutGenerator:
    def __init__(self, num_shards, rate_per_step=0.001, avg_duration=50):
        self.num_shards = num_shards
        self.rate = rate_per_step
        self.avg_duration = avg_duration
        self.active = []

    def step(self, current_time):
        # Stochastic occurrence
        if np.random.random() < self.rate:
            num_affected = np.random.randint(1, self.num_shards // 2 + 1)
            shard_ids = np.random.choice(self.num_shards, num_affected, replace=False)
            duration = np.random.exponential(self.avg_duration)
            severity = np.random.uniform(0.5, 1.0)
            self.active.append(BlackoutEvent(shard_ids, current_time, duration, severity))

        # Cleanup expired
        self.active = [b for b in self.active if current_time < b.start_time + b.duration]

    def is_affected(self, shard_id):
        for b in self.active:
            if shard_id in b.shard_ids:
                return b.severity
        return 0.0
