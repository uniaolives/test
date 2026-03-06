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
echo "Building C++ kernel and Python bindings..."
cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

echo "=== Build complete ==="
