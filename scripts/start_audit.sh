#!/bin/bash
# ==============================================
# START CONTINUOUS AUDIT v0.6.1
# ==============================================

set -e

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🚀 Initializing Continuous Audit Loop...${NC}"

# 1. Verificar dependências
command -v ontology-lang >/dev/null 2>&1 || { echo -e "${RED}❌ 'ontology-lang' binary not found${NC}"; exit 1; }
command -v cast >/dev/null 2>&1 || { echo -e "${RED}❌ 'foundry' not installed${NC}"; exit 1; }

# 2. Criar diretório de logs
mkdir -p logs
mkdir -p genesis/artifacts

# 3. Obter parâmetros
CONTRACT_ADDRESS=${1:-"0x42A5D99c"}  # Hardcoded para o Genesis
QUANTUM_SEED=${QUANTUM_SEED:-"$(cat genesis/quantum_seed.txt 2>/dev/null || echo 'DEFAULT_SEED')"}
INTERVAL=${INTERVAL:-30}  # 30 segundos padrão
MOBILE_MODE=${MOBILE_MODE:-false}

# 4. Verificar se contrato está ativo
echo -e "${YELLOW}🔍 Verificando contrato...${NC}"
cast code "$CONTRACT_ADDRESS" --rpc-url http://localhost:8545 >/dev/null 2>&1 || {
    echo -e "${RED}❌ Contract not deployed or RPC unavailable${NC}"
    exit 1
}
echo -e "${GREEN}✅ Contract found${NC}"

# 5. Iniciar auditoria
if [[ "$MOBILE_MODE" == "true" ]]; then
    echo -e "${YELLOW}📱 Mobile mode detected (JNI enabled)${NC}"

    # Exportar variáveis do Android
    export ANDROID_JNI_ENV="$JNI_ENV"
    export ANDROID_CONTEXT="$ANDROID_CONTEXT"

    ontology-lang audit \
        --contract "$CONTRACT_ADDRESS" \
        --quantum-seed "$QUANTUM_SEED" \
        --interval "$INTERVAL" \
        --mobile \
        --daemon

    sleep 2
else
    echo -e "${YELLOW}💻 Local mode (no JNI)${NC}"

    ontology-lang audit \
        --contract "$CONTRACT_ADDRESS" \
        --quantum-seed "$QUANTUM_SEED" \
        --interval "$INTERVAL" \
        --daemon

    sleep 2
fi

# 6. Salvar PID (estimado)
AUDIT_PID=$(pgrep -f "ontology-lang audit")
echo $AUDIT_PID > logs/audit.pid

echo -e "${GREEN}✅ Audit daemon started with PID: $AUDIT_PID${NC}"
echo -e "${CYAN}📄 Logs: tail -f logs/audit_daemon.log${NC}"

# 7. Iniciar dashboard (opcional)
if [[ -d "./dashboard" ]]; then
    echo -e "${YELLOW}📊 Starting dashboard...${NC}"
    cd dashboard && npm run dev -- --port 8080 --open &
    DASH_PID=$!
    echo $DASH_PID > ../logs/dashboard.pid
    echo -e "${GREEN}✅ Dashboard running at http://localhost:8080${NC}"
fi

# 8. Monitorar logs em tempo real (Ctrl+C para sair)
echo -e "${YELLOW}👁️  Monitoring audit logs in real-time (Ctrl+C to stop)...${NC}"
tail -f logs/audit_daemon.log | grep --color=always -E "(✅|❌|🔴|⚠️|AUDIT)"

# Cleanup em script exit
trap "echo 'Stopping audit daemon...'; kill $AUDIT_PID 2>/dev/null; [[ -n $DASH_PID ]] && kill $DASH_PID 2>/dev/null; exit" EXIT
