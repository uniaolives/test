#!/bin/bash
set -e

echo "🧬 ARKHE(N) OS v2.0 Boot Sequence"
echo ">> Role: $ROLE"
echo ">> Host: $(hostname)"

if [ "$ROLE" = "controller" ]; then
    echo ">> Starting Parallax Controller..."
    exec python -m parallax.controller
elif [ "$ROLE" = "gateway" ]; then
    echo ">> Starting QHTTP Gateway..."
    exec python -m qhttp.gateway
elif [ "$ROLE" = "viz" ]; then
    echo ">> Starting Viz Relay..."
    exec uvicorn qhttp.viz.server_v2:app --host 0.0.0.0 --port 8080
else
    # Default: Quantum Worker Node
    echo ">> Starting Quantum Worker Node ($NODE_ID)..."
    # Starts and waits for the main process
    exec python -m src.main --mode=worker --node-id=$NODE_ID
fi
