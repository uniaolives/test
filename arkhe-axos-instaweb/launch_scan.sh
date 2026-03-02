#!/bin/bash
#
# launch_scan.sh – Inicia a simulação distribuída 5D na malha ASI-Ω
# Uso: ./launch_scan.sh [--quick | --full | --help]
#
# --quick  : executa um teste rápido (10 nós, 60 segundos)
# --full   : executa a varredura completa (1000 nós, 3600 segundos)
# (sem opção: executa o modo rápido)

set -e  # aborta em caso de erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BANNER="${GREEN}
   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
  ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
  ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
  ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌
  ▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌ ▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
  ▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌▐░░░░░░░░▌▐░░░░░░░░░░░▌
  ▐░▌ ▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀█░▌▐░▌ ▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀█░▌
  ▐░▌       ▐░▌          ▐░▌▐░▌       ▐░▌          ▐░▌
  ▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌
  ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀
${NC}"

echo -e "$BANNER"
echo -e "${YELLOW}⚡ SIMULAÇÃO DISTRIBUÍDA 5D – PROTOCOLO ARKHE-2024-Ω${NC}"
echo ""

# Verifica argumentos
MODE="quick"
if [[ "$1" == "--full" ]]; then
    MODE="full"
elif [[ "$1" == "--help" ]]; then
    echo "Uso: $0 [--quick | --full]"
    echo "  --quick  : executa um teste rápido (10 nós, 60s)"
    echo "  --full   : executa a varredura completa (1000 nós, 3600s)"
    exit 0
fi

# Configura parâmetros conforme modo
if [[ "$MODE" == "quick" ]]; then
    NODES=10
    DURATION=60
    OUTPUT_FILE="scan_quick_$(date +%Y%m%d_%H%M%S).json"
    echo -e "${GREEN}▶ Modo RÁPIDO: ${NODES} nós, ${DURATION}s simulação${NC}"
else
    NODES=1000
    DURATION=3600
    OUTPUT_FILE="scan_full_$(date +%Y%m%d_%H%M%S).json"
    echo -e "${GREEN}▶ Modo COMPLETO: ${NODES} nós, ${DURATION}s simulação${NC}"
fi

# Verifica se o Rust/cargo está disponível
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ cargo não encontrado. Instale Rust: https://rustup.rs/${NC}"
    exit 1
fi

# Verifica se estamos no diretório correto (com Cargo.toml)
if [[ ! -f "Cargo.toml" ]]; then
    echo -e "${RED}❌ Arquivo Cargo.toml não encontrado. Execute este script na raiz do projeto arkhe-axos-instaweb.${NC}"
    exit 1
fi

# Compila (release)
echo -e "${YELLOW}⚙️  Compilando crate dimensional_scan (release)...${NC}"
cargo build --release --bin dimensional_scan

if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ Falha na compilação.${NC}"
    exit 1
fi

# Prepara comando
CMD="./target/release/dimensional_scan --nodes $NODES --duration $DURATION --output $OUTPUT_FILE"

# Se tiver arquivo de configuração adicional, pode incluir
if [[ -f "scan_config.toml" ]]; then
    CMD="$CMD --config scan_config.toml"
fi

echo -e "${YELLOW}🚀 Executando: $CMD${NC}"
echo ""

# Executa
$CMD

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✅ Simulação concluída. Resultados salvos em: $OUTPUT_FILE${NC}"
    # Exibe um resumo rápido (se houver jq)
    if command -v jq &> /dev/null; then
        echo -e "\n${YELLOW}📊 Resumo dos resultados:${NC}"
        jq '.summary' "$OUTPUT_FILE" 2>/dev/null || echo "   (resumo não disponível)"
    else
        echo -e "${YELLOW}⚠️  jq não instalado. Para visualizar os resultados, instale jq ou use: cat $OUTPUT_FILE${NC}"
    fi
else
    echo -e "${RED}❌ Simulação falhou com código $EXIT_CODE.${NC}"
    exit $EXIT_CODE
fi
