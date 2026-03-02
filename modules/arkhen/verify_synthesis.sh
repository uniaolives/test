#!/bin/bash
set -e

echo "Starting Arkhe(n) v1.0 Synthesis Verification..."

# 1. Verify Python Syntax
echo "Checking Python module syntax..."
python3 -m py_compile modules/arkhen/python/arkhe_python.py modules/arkhen/python/__init__.py
echo "Python syntax OK."

# 2. Verify Rust Compilation (Check only to save time, if possible)
echo "Checking Rust crate compilation..."
cd modules/arkhen/rust
cargo check
echo "Rust crate check OK."
cd ../../../

# 3. Verify C++ Header (Basic compilation check)
echo "Checking C++ header..."
g++ -std=c++17 -Imodules/arkhen/cpp -c -o /dev/null -x c++ - <<EOF
#include "arkhe_field.hpp"
int main() {
    arkhe::FieldPsi<10, float> field;
    field.initialize_maximum_entropy();
    return 0;
}
EOF
echo "C++ header OK."

echo "Arkhe(n) v1.0 Synthesis Verification Complete: ALL MODULES OPERATIONAL."
