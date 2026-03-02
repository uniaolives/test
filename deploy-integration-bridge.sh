#!/bin/bash
# deploy-integration-bridge.sh
echo "🔗 Deploying Kirchhoff-SASC Integration Bridge..."
python3 -m integration.kirchhoff_sasc_integration &
echo "✅ Integration Bridge Active."
