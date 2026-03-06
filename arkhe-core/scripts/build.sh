#!/bin/bash
set -e

echo "=== Arkhe(n) Core Build System ==="

# Directories
mkdir -p build
mkdir -p data

# Build Rust
echo "Building Rust constitutional layer..."
cd rust
cargo build --release
cd ..
cp ../target/release/libarkhen_rust.so build/

# Build C++
echo "Building C++ kernel..."
cd build
cmake ..
make -j$(nproc)
cd ..

echo "=== Build complete ==="
