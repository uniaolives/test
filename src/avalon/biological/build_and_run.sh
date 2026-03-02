#!/bin/bash
# build_and_run.sh

echo "🔱 CONSTRUINDO SIMULADOR NEURAL QUÂNTICO AVALON"
echo "=============================================="

# 1. Criar diretório de build
mkdir -p build_cpp
cd build_cpp

# 2. Configurar CMake
echo "Configurando CMake..."
cmake ..

# 3. Compilar
echo "Compilando..."
make

# 4. Executar teste
echo ""
echo "🧪 EXECUTANDO TESTE"
./avalon_test

echo ""
echo "✅ Compilação e execução de teste concluídas!"
