#!/bin/bash

echo "=== Aizawa High Performance Benchmark ==="

# 1. Python Numba
echo ""
echo "--- Python Numba ---"
python3 aizawa_numba.py

# 2. C++ AVX2 + OpenMP
echo ""
echo "--- C++ AVX2 + OpenMP ---"
g++ -O3 -mavx2 -fopenmp aizawa_avx.cpp -o aizawa_avx
if [ -f ./aizawa_avx ]; then
    ./aizawa_avx
else
    echo "C++ compilation failed."
fi

# 3. Rust Rayon
echo ""
echo "--- Rust Rayon ---"
cargo build --release
if [ -f ./target/release/aizawa_high_performance ]; then
    ./target/release/aizawa_high_performance
else
    echo "Rust build failed."
fi

echo ""
echo "=== Benchmark Finished ==="
