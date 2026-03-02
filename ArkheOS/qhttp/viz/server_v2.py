# qhttp/viz/server_v2.py
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Set
from dataclasses import dataclass, asdict
from datetime import datetime

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AQV.Relay")

# --- Prometheus Metrics ---
WS_CONNECTIONS = Counter('quantum_ws_connections_total', 'Total WebSocket connections')
WS_MESSAGES = Counter('quantum_ws_messages_total', 'Messages sent', ['type'])
STATE_UPDATE_DURATION = Histogram('quantum_state_update_seconds', 'Time spent updating state')
EVENTS_RECEIVED = Counter('quantum_events_total', 'Quantum events received', ['channel'])

@dataclass
class QuantumState:
    nodes: Dict
    links: list
    timestamp: float

    def to_dict(self):
        return {
            'nodes': self.nodes,
            'links': self.links,
            'timestamp': self.timestamp
        }

class QuantumStateManager:
    """Atomic state management with per-client queues."""
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self._state_cache = QuantumState({}, [], 0.0)
        self._lock = asyncio.Lock()
        self._subscribers: Set[asyncio.Queue] = set()

    async def subscribe(self) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(queue)
            await queue.put(self._state_cache)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue):
        async with self._lock:
            self._subscribers.discard(queue)

    @STATE_UPDATE_DURATION.time()
    async def update_state(self):
        """Fetch current state from Redis and notify subscribers."""
        try:
            pipe = self.redis.pipeline()
            pipe.keys("entanglement:entangle:*")
            pipe.hgetall("quantum:metrics")
            results = await pipe.execute()
            keys = results[0]
            metrics = results[1]

            active_links = []
            for key in keys:
                # key is a string because decode_responses=True
                data = await self.redis.hgetall(key)
                if data.get('active') == '1':
                    active_links.append({
                        "id": key,
                        "u": data.get('node_a', 'unknown'),
                        "v": data.get('node_b', 'unknown'),
                        "state": int(data.get('bell_type', 0)),
                        "fidelity": float(data.get('fidelity', 0.99))
                    })

            # Node metrics (simulated with real Redis data)
            nodes = {
                "arkhe-node-1": {"load": 0.4, "coherence": 0.99, "agents": 150},
                "arkhe-node-2": {"load": 0.6, "coherence": 0.97, "agents": 500},
                "arkhe-node-3": {"load": 0.2, "coherence": 0.995, "agents": 1000}
            }

            new_state = QuantumState(
                nodes=nodes,
                links=active_links,
                timestamp=asyncio.get_event_loop().time()
            )

            async with self._lock:
                self._state_cache = new_state
                dead_subs = set()
                for queue in self._subscribers:
                    try:
                        queue.put_nowait(new_state)
                    except asyncio.QueueFull:
                        dead_subs.add(queue)
                for dead in dead_subs:
                    self._subscribers.discard(dead)

        except Exception as e:
            logger.error(f"State update failed: {e}")

class EventStream:
    """Redis event stream with auto-reconnection."""
    def __init__(self, redis_url: str, state_manager: QuantumStateManager):
        self.redis_url = redis_url
        self.state_manager = state_manager
        self._running = False
        self._task = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        retry_delay = 1.0
        max_delay = 30.0
        while self._running:
            try:
                redis = aioredis.from_url(self.redis_url, decode_responses=True)
                pubsub = redis.pubsub()
                await pubsub.subscribe("quantum:events", "qec:syndrome")
                logger.info("Event stream connected")
                retry_delay = 1.0

                async for message in pubsub.listen():
                    if not self._running:
                        break
                    if message["type"] == "message":
                        await self._handle_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event stream error: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)

    async def _handle_message(self, message: dict):
        try:
            data = json.loads(message["data"])
            channel = message["channel"]
            EVENTS_RECEIVED.labels(channel=channel).inc()

            event = {
                "type": "EVENT",
                "channel": channel,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }

            if channel == "quantum:events" and data.get("type") in ["ENTANGLE", "COLLAPSE"]:
                await self.state_manager.update_state()

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON: {message['data']}")

# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global state_manager, event_stream
    redis = aioredis.from_url(
        "redis://redis:6379",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True
    )
    state_manager = QuantumStateManager(redis)
    event_stream = EventStream("redis://redis:6379", state_manager)
    await event_stream.start()

    async def state_updater():
        while True:
            await state_manager.update_state()
            await asyncio.sleep(0.2)  # 5 Hz
    updater_task = asyncio.create_task(state_updater())

    yield

    updater_task.cancel()
    await event_stream.stop()
    await redis.close()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="qhttp/viz/static"), name="static")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.websocket("/ws/quantum_stream")
async def quantum_websocket(websocket: WebSocket):
    await websocket.accept()
    WS_CONNECTIONS.inc()
    queue = await state_manager.subscribe()

    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            try:
                state = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_json({
                    "type": "SNAPSHOT",
                    **state.to_dict()
                })
                WS_MESSAGES.labels(type='snapshot').inc()
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "HEARTBEAT"})
                WS_MESSAGES.labels(type='heartbeat').inc()
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        WS_CONNECTIONS.dec()
        await state_manager.unsubscribe(queue)
