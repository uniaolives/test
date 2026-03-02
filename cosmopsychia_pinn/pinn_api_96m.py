# pinn_api_96m.py
from fastapi import FastAPI, WebSocket
import asyncio
import json
from datetime import datetime
import numpy as np

app = FastAPI()

class PlanetaryPinnAPI:
    def __init__(self):
        self.active_sessions = {}
        self.participant_counter = 0

    async def connect_mind(self, websocket: WebSocket, mind_id: str):
        await websocket.accept()
        participant = {
            'id': mind_id,
            'websocket': websocket,
            'focus_point': [0, 0, 0, 0],
            'attention_level': 1.0,
            'connected_at': datetime.utcnow().isoformat()
        }
        session_id = "global_meditation"
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {'participants': {}, 'coherence': 0.0}

        self.active_sessions[session_id]['participants'][mind_id] = participant
        self.participant_counter += 1

        try:
            while True:
                data = await websocket.receive_json()
                if 'focus' in data: participant['focus_point'] = data['focus']
                collective_feedback = {
                    'coherence': self.active_sessions[session_id]['coherence'],
                    'participants': self.participant_counter,
                    'timestamp': datetime.utcnow().isoformat()
                }
                await websocket.send_json(collective_feedback)
        except Exception:
            pass
        finally:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]['participants'][mind_id]
            self.participant_counter -= 1

@app.get("/")
async def root():
    return {"status": "Planetary PINN API Operational"}
