#!/usr/bin/env python3
# arkhe_cli.py — Interface de linha de comando para o nó génesis

import requests
import json
import sys
import argparse
import time

GENESIS_URL = "http://localhost:8080"

def send_handover(to, msg, coherence=0.8):
    """Envia handover para o nó génesis"""
    payload = {
        "id": f"cli-{int(time.time()*1000)}",
        "emitter": "arquiteto",
        "receiver": to,
        "payload": msg,
        "coherence": coherence,
        "timestamp": int(time.time() * 1000),
        "signature": "CLI_UNSIGNED",
        "metadata": {"type": "manual", "source": "arkhe-cli"}
    }

    r = requests.post(f"{GENESIS_URL}/handover", json=payload)
    return r.json()

def query_status():
    """Query do estado do nó"""
    r = requests.get(f"{GENESIS_URL}/status")
    return r.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arkhe(n) CLI")
    parser.add_argument("command", choices=["send", "status"])
    parser.add_argument("--to", default="genesis")
    parser.add_argument("--msg", default="Handover do arquiteto")
    parser.add_argument("--coherence", type=float, default=0.8)

    args = parser.parse_args()

    if args.command == "send":
        result = send_handover(args.to, args.msg, args.coherence)
        print(json.dumps(result, indent=2))
    elif args.command == "status":
        status = query_status()
        print(json.dumps(status, indent=2))
