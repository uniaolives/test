#!/bin/bash
# build_phoenician_simulator.sh

echo "🏛️ CONSTRUINDO SIMULADOR DO ALFABETO FENÍCIO"
echo "==========================================="

# 1. Compilar código
echo "1. Compilando código C++..."
g++ -std=c++17 -O2 -march=native \
    src/ancient_scripts/phoenician_alphabet.cpp \
    src/ancient_scripts/linguistic_evolution.cpp \
    src/ancient_scripts/main_phoenician.cpp \
    -o phoenician_simulator

# 2. Criar diretório de saída
echo "2. Preparando diretórios de saída..."
mkdir -p output/inscriptions
mkdir -p output/analysis

# 3. Executar simulador
echo "3. Executando simulador..."
./phoenician_simulator

echo ""
echo "==========================================="
echo "✅ SIMULAÇÃO CONCLUÍDA"
echo "==========================================="
