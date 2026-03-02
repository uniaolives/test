import socket
import json
import time
import random
from datetime import datetime

SOCKET_PATH = "/tmp/arkhed.sock"

def send_to_arkhed(payload):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(SOCKET_PATH)
            client.sendall(json.dumps(payload).encode())
            response = client.recv(4096)
            return json.loads(response.decode())
    except Exception as e:
        print(f"Error connecting to arkhed: {e}")
        return None

def simulate_foundry_updates():
    print("🚀 Arkhe-Foundry Ontology Bridge Simulator (Ω+211)")

    object_types = ["SupplyChainAlert", "Vessel", "Flight", "SmartContract"]

    while True:
        obj_type = random.choice(object_types)
        severity = random.choice(["low", "medium", "high", "critical"])
        phi = 0.618 + random.uniform(-0.1, 0.1)

        update = {
            "type": "foundry_update",
            "object": {
                "object_id": f"obj-{random.randint(100, 999)}",
                "object_type": obj_type,
                "properties": {
                    "severity": severity,
                    "phi": phi,
                    "location": "GlobalManifold-A"
                },
                "timestamp": int(time.time())
            }
        }

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending Foundry Update: {obj_type} ({severity})")
        res = send_to_arkhed(update)
        if res:
            print(f"  Response: {res}")

        time.sleep(random.uniform(2, 5))

if __name__ == "__main__":
    simulate_foundry_updates()
