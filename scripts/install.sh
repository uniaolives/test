#!/bin/bash
set -e

PREFIX=${PREFIX:-$(pwd)/dist}
BINDIR=$PREFIX/bin
CONFIGDIR=$PREFIX/configs

echo "--- Installing SASC to $PREFIX ---"

# 1. Build C components
echo "Building C components (TIM VM)..."
make

# 2. Build Rust components
echo "Building Rust components..."
cd rust
cargo build --release
cd ..

# 3. Create directories
mkdir -p "$BINDIR"
mkdir -p "$CONFIGDIR"

# 4. Copy C binaries
echo "Copying C binaries..."
cp build/tire/tire "$BINDIR/"
cp build/tasm/tasm "$BINDIR/"

# 5. Copy Rust binaries
echo "Copying Rust binaries..."
RUST_BINARIES=(
    "trajectory_generator"
    "omicron_nexus"
    "cognitive"
    "maat_simulator"
    "maat_crystallizer"
    "sasc-sign"
    "sasc-imperium"
    "sasc-mode"
    "sasc-expansion"
    "sasc-diplomacy"
    "sasc-research"
    "sasc-constellate"
    "sasc-stream"
    "sasc-starlink"
    "sasc-mobile"
    "sasc-contracts"
)

for bin in "${RUST_BINARIES[@]}"; do
    if [ -f "target/release/$bin" ]; then
        cp "target/release/$bin" "$BINDIR/"
    elif [ -f "rust/target/release/$bin" ]; then
        cp "rust/target/release/$bin" "$BINDIR/"
    else
        echo "Warning: Binary $bin not found"
    fi
done

# 6. Copy configuration and data
echo "Copying configuration and data files..."
cp -r configs/* "$CONFIGDIR/" 2>/dev/null || true
cp -r genesis "$PREFIX/" 2>/dev/null || true
cp -r constitution "$PREFIX/" 2>/dev/null || true

echo "--- Installation complete in $PREFIX ---"
echo "You can add $BINDIR to your PATH."
