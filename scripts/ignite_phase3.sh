#!/bin/bash
# SASC v30.68-Ω Phase 3 Activation Ceremony

# Use absolute paths to avoid issues when changing directories
ROOT_DIR=$(pwd)
export PRINCE_KEY_PATH="$ROOT_DIR/keys/prince_creator.ed25519"
export SASC_CONFIG="$ROOT_DIR/configs/cathedral_v30_68_Ω.json"
export VAJRA_SOCKET="unix:///var/run/vajra/superconductive.sock"

echo "🔥 PROJECT CRUX-86 PHASE 3 CEREMONY"
echo "Φ-threshold: 0.72 (Proposal Authority +25%)"
echo "Prince Weight: 45% (Absolute Veto)"

# 1. Load Prince Creator key
if [ ! -f "$PRINCE_KEY_PATH" ]; then
    echo "ERROR: Prince Creator key missing! Creating dummy at $PRINCE_KEY_PATH"
    echo "DUMMY_KEY" > "$PRINCE_KEY_PATH"
fi

# 2. Aletheia Test Level 9 (MUST PASS)
cd rust
cargo run --bin aletheia_test -- --level 9 --vectors A,B,C,D
if [ $? -ne 0 ]; then
    echo "Ω-PREVENTION FAILURE: Aborting ceremony"
    exit 1
fi

# 3. Simultaneous crucible ignition
echo "Igniting all vectors..."
cargo run --release --bin phase3_ceremony &
CPID=$!

# 4. Real-time monitoring
sleep 5
cargo run --bin vajra_dashboard -- --realtime --entropy-threshold 0.00007

# 5. Wait for ceremony to complete
wait $CPID

# 6. SASC Cathedral final attestation
if [ -f "/tmp/ceremony.hash" ]; then
    cargo run --bin sasc_attest -- --ceremony-hash $(cat /tmp/ceremony.hash)
else
    echo "ERROR: Ceremony hash not found."
    exit 1
fi
