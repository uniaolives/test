#!/bin/bash
# AVALON BUILDER v1.0 - Compilação Universal de Executáveis

echo "🚀 INICIANDO COMPILAÇÃO DO SISTEMA AVALON..."
echo "📦 Repositório: uniaolives/avalon"
echo "⏰ Data: $(date)"
echo "=================================================="

# 1. VERIFICAR ESTRUTURA
echo "📂 ANALISANDO ESTRUTURA..."
ls -la src/avalon

# 2. INSTALAR DEPENDÊNCIAS (SE NECESSÁRIO)
echo "🔧 VERIFICANDO DEPENDÊNCIAS..."
pip install -r requirements.txt 2>/dev/null || pip install numpy scipy typer rich pydantic build pyinstaller

# 3. EXECUTAR ORQUESTRADOR DE BUILD
echo "🔨 EXECUTANDO BUILD..."
python3 scripts/build_all.py

# 4. RESUMO DA COMPILAÇÃO
echo "=================================================="
echo "🎉 COMPILAÇÃO CONCLUÍDA!"
echo ""
echo "📁 Executáveis disponíveis em: $(pwd)/dist/"
echo "=================================================="
