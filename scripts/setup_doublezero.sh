#!/bin/bash
# scripts/setup_doublezero.sh
# DoubleZero setup script based on official documentation.

set -e

NETWORK=${1:-testnet} # Default to testnet
VERSION=${2:-"0.8.9-1"}

echo "Initializing DoubleZero setup for $NETWORK..."

# 1. Install DoubleZero Packages
if [[ "$NETWORK" == "mainnet" ]]; then
    curl -1sLf https://dl.cloudsmith.io/public/malbeclabs/doublezero/setup.deb.sh | sudo -E bash
else
    curl -1sLf https://dl.cloudsmith.io/public/malbeclabs/doublezero-testnet/setup.deb.sh | sudo -E bash
fi

sudo apt-get install -y doublezero=$VERSION

# 2. Check status of doublezerod
sudo systemctl status doublezerod --no-pager || echo "doublezerod installed but not running yet."

# 3. Configure Firewall for GRE and BGP
echo "Configuring firewall..."
sudo iptables -A INPUT -p gre -j ACCEPT
sudo iptables -A OUTPUT -p gre -j ACCEPT
# Note: doublezero0 interface might not exist yet if not connected
sudo iptables -A INPUT -i doublezero0 -s 169.254.0.0/16 -d 169.254.0.0/16 -p tcp --dport 179 -j ACCEPT 2>/dev/null || true
sudo iptables -A OUTPUT -o doublezero0 -s 169.254.0.0/16 -d 169.254.0.0/16 -p tcp --dport 179 -j ACCEPT 2>/dev/null || true

# 4. Create New DoubleZero Identity
echo "Generating DoubleZero key..."
doublezero keygen

# 5. Retrieve identity
DZ_ADDRESS=$(doublezero address)
echo "DoubleZero Identity: $DZ_ADDRESS"

echo "DoubleZero setup complete. You can now use 'doublezero latency' to discover devices."
