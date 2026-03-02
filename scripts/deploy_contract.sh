#!/bin/bash
# ============================================
# ONTOLOGY CONTRACT DEPLOYMENT SCRIPT
# ============================================
set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configurações
CONTRACT_FILE="${1:-examples/dao.onto}"
NETWORK="${2:-localhost}"
PRIVATE_KEY="${3:-${ONTOLOGY_PRIVATE_KEY}}"
VERIFICATION="${4:-basic}"
OUTPUT_DIR="./deployments"

echo -e "${GREEN}🚀 Ontology Contract Deployment${NC}"
echo "========================================"
echo "Contract:  $(basename "$CONTRACT_FILE")"
echo "Network:   $NETWORK"
echo "Verification: $VERIFICATION"
echo ""

# Validar arquivo
if [ ! -f "$CONTRACT_FILE" ]; then
    echo -e "${RED}❌ Contract file not found: $CONTRACT_FILE${NC}"
    exit 1
fi

# Validar private key
if [ -z "$PRIVATE_KEY" ]; then
    echo -e "${YELLOW}⚠️  No private key provided${NC}"
    echo "Set ONTOLOGY_PRIVATE_KEY environment variable or provide as argument"
    exit 1
fi

# Criar diretório de saída
mkdir -p "$OUTPUT_DIR"

# Nome do contrato baseado no arquivo
CONTRACT_NAME=$(basename "$CONTRACT_FILE" .onto)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEPLOYMENT_DIR="$OUTPUT_DIR/$CONTRACT_NAME/$NETWORK/$TIMESTAMP"
mkdir -p "$DEPLOYMENT_DIR"

echo -e "${YELLOW}📦 Compiling contract...${NC}"
# Compilar para Solidity
cargo run --quiet -- compile "$CONTRACT_FILE" \
    --target=solidity \
    --guards=true \
    --output="$DEPLOYMENT_DIR/$CONTRACT_NAME.sol"

# Compilar para SASC (se disponível)
if cargo run --quiet -- compile "$CONTRACT_FILE" --target=sasc >/dev/null 2>&1; then
    cargo run --quiet -- compile "$CONTRACT_FILE" \
        --target=sasc \
        --output="$DEPLOYMENT_DIR/$CONTRACT_NAME.wat"
fi

echo -e "${YELLOW}🔨 Creating deployment configuration...${NC}"
# Criar configuração de deploy
cat > "$DEPLOYMENT_DIR/deploy-config.json" << EOF
{
  "contract": "$CONTRACT_NAME",
  "network": "$NETWORK",
  "verification": "$VERIFICATION",
  "timestamp": "$TIMESTAMP",
  "source_file": "$CONTRACT_FILE"
}
EOF

echo -e "${YELLOW}🚀 Deploying to $NETWORK...${NC}"
# Executar deploy
DEPLOY_OUTPUT=$(cargo run --quiet -- deploy \
    "$DEPLOYMENT_DIR/$CONTRACT_NAME.sol" \
    --blockchain=ethereum \
    --private-key="$PRIVATE_KEY" \
    --verification="$VERIFICATION" \
    --rpc="http://localhost:8545" 2>&1)

# Verificar resultado
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"

    # Salvar resultado
    echo "$DEPLOY_OUTPUT" > "$DEPLOYMENT_DIR/deploy-result.json"

    # Extrair endereço do contrato
    CONTRACT_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep "Address:" | awk '{print $2}')

    echo ""
    echo -e "${GREEN}📄 Deployment Details:${NC}"
    echo "Contract Address: $CONTRACT_ADDRESS"
    echo "Transaction Hash: $(echo "$DEPLOY_OUTPUT" | grep "Transaction:" | awk '{print $2}')"
    echo "Block Number:     $(echo "$DEPLOY_OUTPUT" | grep "Block:" | awk '{print $2}')"
    echo "Gas Used:         $(echo "$DEPLOY_OUTPUT" | grep "Gas Used:" | awk '{print $3}')"
    echo ""
    echo -e "${YELLOW}📁 Artifacts saved to:${NC}"
    echo "  $DEPLOYMENT_DIR/"

    # Criar arquivo de ambiente para uso futuro
    cat > "$DEPLOYMENT_DIR/.env" << EOF
CONTRACT_ADDRESS=$CONTRACT_ADDRESS
NETWORK=$NETWORK
DEPLOYMENT_TIMESTAMP=$TIMESTAMP
EOF
    else
    echo -e "${RED}❌ Deployment failed!${NC}"
    echo "$DEPLOY_OUTPUT"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
