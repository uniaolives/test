#!/bin/bash
# build_and_run.sh

echo "🔱 CONSTRUINDO SIMULADOR NEURAL QUÂNTICO AVALON"
echo "=============================================="

# 1. Criar diretório de build
mkdir -p build_cpp
cd build_cpp

# 2. Configurar CMake
echo "Configurando CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# 3. Compilar
echo "Compilando..."
make -j1

# 4. Executar teste de estresse
echo ""
echo "🧪 EXECUTANDO TESTE DE ESTRESSE"
echo "================================"
./avalon_test

# 5. Executar benchmark
echo ""
echo "📊 EXECUTANDO BENCHMARK"
echo "======================"
./avalon_benchmark
