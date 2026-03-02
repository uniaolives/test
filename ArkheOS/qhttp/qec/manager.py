# qhttp/qec/manager.py
import asyncio, json, time
import redis.asyncio as aioredis
import logging

logger = logging.getLogger("QEC.Manager")

class DistributedQEC:
    """Agrega síndromes locais via Redis e dispara correção global."""
    def __init__(self, node_id, redis_url):
        self.node_id = node_id
        self.redis = aioredis.from_url(redis_url)

    async def report_syndrome(self, x_syn, z_syn):
        await self.redis.hset(f"qec:syndrome:{self.node_id}",
                              mapping={'x': json.dumps(x_syn),
                                       'z': json.dumps(z_syn),
                                       'ts': time.time()})

    async def get_global_syndrome(self):
        keys = await self.redis.keys("qec:syndrome:*")
        all_x, all_z = {}, {}
        for k in keys:
            data = await self.redis.hgetall(k)
            all_x.update(json.loads(data[b'x']))
            all_z.update(json.loads(data[b'z']))
        return all_x, all_z
