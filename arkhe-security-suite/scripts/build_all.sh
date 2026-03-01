#!/bin/bash
set -e
echo "Building Arkhe(n) Security Suite (Ω+209-223)..."

cd "$(dirname "$0")/.."

echo "Building Rust Workspace..."
cargo build

echo "Building Go components..."
(cd comms/quic && go build .)
(cd command/emergency_authority && go build .)

echo "Build complete."
