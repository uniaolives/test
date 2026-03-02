# sasc-core/orchestrator.py
import asyncio
import json
import os
import nats
from nats.js.api import StreamConfig, RetentionPolicy

class Orchestrator:
    def __init__(self):
        self.nodes = {}
        self.tasks = {}

    async def run(self):
        nats_url = os.environ.get("NATS_URL", "nats://nats:4222")
        nc = await nats.connect(nats_url)
        js = nc.jetstream()

        # Setup streams
        try:
            await js.add_stream(name="nodes", subjects=["nodes.register"])
            await js.add_stream(name="tasks", subjects=["tasks.*", "tasks.results"])
        except Exception as e:
            print(f"ℹ️ Stream already exists or error: {e}")

        # Subscribe to node registration
        await js.subscribe("nodes.register", cb=self.register_node)

        # Subscribe to task results
        await js.subscribe("tasks.results", cb=self.handle_result)

        print("🏛️ SASC Orchestrator (Brain) is running...")
        while True:
            await asyncio.sleep(1)

    async def register_node(self, msg):
        data = json.loads(msg.data.decode())
        node_id = data.get('node_id', 'unknown')
        self.nodes[node_id] = data
        print(f"✅ Node registered: {node_id} ({data.get('node_type', 'generic')})")
        await msg.ack()

    async def handle_result(self, msg):
        data = json.loads(msg.data.decode())
        task_id = data.get('task_id', 'unknown')
        print(f"📥 Task result received: {task_id} from {data.get('node_id', 'unknown')}")
        self.tasks[task_id] = data.get('result')
        await msg.ack()

if __name__ == "__main__":
    orchestrator = Orchestrator()
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass
