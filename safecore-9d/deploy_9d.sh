#!/bin/bash
# SAFECORE-9D DEPLOYMENT SCRIPT
# Execute: ./deploy_9d.sh [--dev|--prod]

set -euo pipefail

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Configurações
MODE="${1:---dev}"
RUST_FLAGS=""
CARGO_FLAGS=""

case "$MODE" in
    --dev)
        log "Modo: Desenvolvimento"
        RUST_FLAGS=""
        CARGO_FLAGS=""
        ;;
    --prod)
        log "Modo: Produção"
        RUST_FLAGS="-C opt-level=3 -C target-cpu=native -C lto=fat"
        CARGO_FLAGS="--release"
        ;;
    *)
        error "Modo inválido: $MODE"
        echo "Uso: $0 [--dev|--prod]"
        exit 1
        ;;
esac

# ============================ VALIDAÇÃO ============================
log "Validando ambiente..."

# Verificar Rust
if ! command -v rustc > /dev/null 2>&1; then
    error "Rust não encontrado"
    echo "Instale com: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
success "Rust: $(rustc --version | awk '{print $2}')"

# Verificar Cargo
if ! command -v cargo > /dev/null 2>&1; then
    error "Cargo não encontrado"
    exit 1
fi
success "Cargo disponível"

# ============================ COMPILAÇÃO ============================
log "Compilando SafeCore-9D..."

export RUSTFLAGS="$RUST_FLAGS"
export CARGO_PROFILE_RELEASE_LTO="fat"
export CARGO_PROFILE_RELEASE_OPT_LEVEL="3"

if [ "$MODE" = "--prod" ]; then
    log "Compilação de produção otimizada..."
    cargo build --release $CARGO_FLAGS
    BINARY_PATH="./target/release/safecore-9d"
else
    log "Compilação de desenvolvimento..."
    cargo build $CARGO_FLAGS
    BINARY_PATH="./target/debug/safecore-9d"
fi

if [ ! -f "$BINARY_PATH" ]; then
    error "Falha na compilação"
    exit 1
fi
success "Compilação completa: $BINARY_PATH"

# ============================ CONFIGURAÇÃO ============================
log "Configurando ambiente 9D..."

# Criar diretórios de runtime
mkdir -p ~/.safecore/run
mkdir -p ~/.safecore/logs

# Configurar variáveis
export SAFECORE_9D_MODE="$MODE"
export SAFECORE_9D_VERSION="9.0.0"
export SAFECORE_PHI_TARGET="1.030"
export SAFECORE_TAU_MAX="1.35"
export SAFECORE_SHARD_ID="alpha"
export SAFECORE_DIMENSIONS="9"

# Gerar ID de instância
INSTANCE_ID=$(date +%s%N | sha256sum | head -c 16)
export SAFECORE_INSTANCE_ID="$INSTANCE_ID"
success "Instance ID: $INSTANCE_ID"

# ============================ INICIALIZAÇÃO ============================
log "Inicializando SafeCore-9D..."

# Verificar porta disponível
check_port() {
    local port=$1
    if command -v ss > /dev/null 2>&1; then
        ss -tuln 2>/dev/null | grep -q ":$port " && echo "ocupada" || echo "livre"
    else
        echo "livre"
    fi
}

# Encontrar portas livres
DASHBOARD_PORT=9050
while [ "$(check_port $DASHBOARD_PORT)" = "ocupada" ]; do
    DASHBOARD_PORT=$((DASHBOARD_PORT + 1))
done

METRICS_PORT=9100
while [ "$(check_port $METRICS_PORT)" = "ocupada" ]; do
    METRICS_PORT=$((METRICS_PORT + 1))
done

export SAFECORE_DASHBOARD_PORT="$DASHBOARD_PORT"
export SAFECORE_METRICS_PORT="$METRICS_PORT"

# ============================ EXECUÇÃO ============================
log "Iniciando sistema..."

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   SAFECORE-9D v9.0.0                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Modo:          $(printf '%-40s' "$MODE") ║"
echo "║  Instância:     $(printf '%-40s' "$INSTANCE_ID") ║"
echo "║  Dimensões:     9                                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Dashboard:     http://localhost:$DASHBOARD_PORT              ║"
echo "║  Métricas:      http://localhost:$METRICS_PORT               ║"
echo "║  Logs:          ~/.safecore/logs/ ou /var/log/safecore-9d   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
log "Pressione Ctrl+C para encerrar..."

# Executar o binário
"$BINARY_PATH"
