#!/bin/bash
# start-integrated-processes.sh
echo "🔥 Starting Integrated Processes..."
python3 -m sasc_extended.kirchhoff_sasc_system &
echo "✅ All Integrated Processes Started."
