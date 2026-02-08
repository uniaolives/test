#!/bin/bash
# deploy_system.sh - Setup and execute Phoenician Script Simulator

set -e

echo "🏗️  DEPLOYING PHOENICIAN EVOLUTION SYSTEM"
echo "=========================================="

# 1. Create directory structure
mkdir -p bin output/inscriptions output/analysis web

# 2. Build the system
echo "2. Building C++ core..."
cd src/ancient_scripts
make all
make bin/test_evolution || g++ -std=c++17 -O3 -I. phoenician_alphabet.cpp linguistic_evolution.cpp evolution_analysis.cpp ../../tests/test_linguistic_evolution.cpp -o ../../bin/test_evolution
cd ../..

# 3. Run tests
echo "3. Running unit tests..."
./bin/test_evolution

# 4. Run full analysis
echo "4. Executing full simulation..."
./bin/phoenician_simulator

# 5. Move visualizers
echo "5. Deploying web visualizers..."
cp src/ancient_scripts/*.html web/

echo "=========================================="
echo "🎉 DEPLOYMENT COMPLETE"
echo "Check output/ and web/ for results."
