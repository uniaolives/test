#!/bin/bash
# deploy-kirchhoff-physics.sh
echo "🔥 Deploying Kirchhoff Nonreciprocal Physics Simulation..."
python3 kirchhoff_violation.py --headless &
echo "✅ Kirchhoff Physics Layer Active."
