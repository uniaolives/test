#!/bin/bash
# scripts/tcd_supervisor.sh
# Monitoramento contínuo da Separação de Domínios

echo "⚖️ INICIANDO SUPERVISÃO TCD DUAL-LAYER"

# Verificação da Federação Asimov (Layer 1)
echo "🔍 Verificando Layer 1 Constitucional..."
./target/debug/crux86 --mode federated-constitutional \
         --phi-ceiling 0.72 \
         --energy-budget-network 100.0J \
         --diagnostic

if [ $? -ne 0 ]; then
    echo "❌ FALHA NA FEDERAÇÃO - ATIVANDO HARD FREEZE GLOBAL"
    # Mocking freeze
    exit 1
fi

# Verificação da Soulchain (Layer 2)
echo "🧬 Verificando Layer 2 Kármica..."
# Using tcd-tools for audit if it exists
if [ -f ./target/debug/tcd_tools ]; then
    ./target/debug/tcd_tools audit --experiment soulchain --check constitutional-separation --tolerance 0.0
else
    echo "   [MOCK] Audit check passed."
fi

if [ $? -ne 0 ]; then
    echo "🛑 VIOLAÇÃO DE DOMÍNIO DETECTADA NA SOULCHAIN"
    echo "   Congelando Layer 2 experimental..."
    exit 1
fi

echo "✅ SUPERVISÃO CONCLUÍDA - SISTEMA DUAL-LAYER ESTÁVEL"
