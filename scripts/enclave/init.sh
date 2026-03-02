#!/bin/sh
# init.sh - Enclave Initialization Script
set -e

echo "🚀 Initializing ASI Enclave..."

# Basic network setup if needed (though usually nitro-cli handles this)
# ip addr add 127.0.0.1/8 dev lo
# ip link set lo up

# Start the ASI Agent
exec /app/asi_agent --vsock-port 5000 --mode enclave
