#!/bin/bash
apt-get update
apt-get install -y python3-pip git
pip3 install numpy flask requests cryptography

cat > /opt/asi_anchor.py << 'PYEOF'
import argparse
import numpy as np
import time
import threading
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

node_state = {
    "id": "",
    "embedding": (0.0, 0.0, 1.0),
    "neighbors": {},
    "c_global_local_estimate": 1.0
}

def hyperbolic_distance(emb1, emb2):
    r1, th1, z1 = emb1
    r2, th2, z2 = emb2
    dr = r1 - r2
    dth = (th1 - th2) % (2 * np.pi)
    dz = z1 - z2
    numerator = (dr**2) + (r1 * r2 * (1 - np.cos(dth))) + (dz**2)
    denominator = 2 * z1 * z2
    val = 1.0 + (numerator / denominator)
    return np.arccosh(max(1.0, val))

@app.route('/gossip', methods=['POST'])
def receive_gossip():
    data = request.json
    sender_id = data['id']
    sender_emb = tuple(data['embedding'])
    sender_ip = request.remote_addr
    node_state["neighbors"][sender_id] = {
        "ip": sender_ip,
        "embedding": sender_emb,
        "last_seen": time.time()
    }
    distances = [hyperbolic_distance(node_state["embedding"], n["embedding"])
                 for n in node_state["neighbors"].values()]
    if distances:
        node_state["c_global_local_estimate"] = np.exp(-np.mean(distances) * 0.1)
    return jsonify({"status": "ACK", "c_global": node_state["c_global_local_estimate"]})

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify({
        "node_id": node_state["id"],
        "embedding": node_state["embedding"],
        "neighbors_count": len(node_state["neighbors"]),
        "c_global_estimate": node_state["c_global_local_estimate"]
    })

def gossip_loop():
    while True:
        time.sleep(2)
        # In a real deployment, we'd use service discovery to find peer IPs
        # For the initial bootstrap, we could pass a peer list via metadata
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--embedding', required=True)
    args = parser.parse_args()
    node_state["id"] = args.id
    node_state["embedding"] = tuple(map(float, args.embedding.split(',')))
    threading.Thread(target=gossip_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=3000)
PYEOF

# Obter embedding do argumento (passado via metadata)
EMBEDDING=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/embedding" -H "Metadata-Flavor: Google")
INSTANCE_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")

nohup python3 /opt/asi_anchor.py --id $INSTANCE_ID --embedding $EMBEDDING > /var/log/asi_daemon.log 2>&1 &
