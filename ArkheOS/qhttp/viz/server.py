import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import redis.asyncio as redis
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Arkhe.Viz")

app = FastAPI()

# Servir frontend estático
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()
redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.from_url(redis_url, decode_responses=True)

@app.websocket("/ws/quantum_stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Snapshot do universo quântico
            nodes = ["arkhe-q1", "arkhe-q2"]

            keys = await redis_client.keys("entanglement:entangle:*")
            entanglements = []

            for key in keys:
                data = await redis_client.hgetall(key)
                if data and data.get('active') == '1':
                    entanglements.append({
                        'pair_id': key.split(":")[-1],
                        'node_a': data['node_a'],
                        'agent_a': int(data['agent_a']),
                        'node_b': data['node_b'],
                        'agent_b': int(data['agent_b']),
                        'bell_type': int(data['bell_type'])
                    })

            await websocket.send_json({
                "type": "SNAPSHOT",
                "nodes": nodes,
                "entanglements": entanglements,
                "timestamp": asyncio.get_event_loop().time()
            })

            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    async def collapse_listener():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("quantum:events")
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                await manager.broadcast({
                    "type": "EVENT",
                    "event_type": "COLLAPSE",
                    "data": data
                })
    asyncio.create_task(collapse_listener())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
