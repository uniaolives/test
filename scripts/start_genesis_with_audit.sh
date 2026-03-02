#!/bin/bash
set -e

echo "========================================="
echo "   GENESIS DAO + CONTINUOUS AUDIT v1.0   "
echo "========================================="

# Configurações
RPC_URL="http://localhost:8545"
CONTRACT_ADDRESS="0x5FbDB2315678afecb367f032d93F642f64180aa3"
QUANTUM_SEED="7f3b2a1c9e8d5f4a2b3c1d0e9f8a7b6c5d4e3f2a1b0c9e8d7f6a5b4c3d2e1f0a"
LOG_DIR="./genesis/logs"
mkdir -p $LOG_DIR
mkdir -p logs

# 1. Iniciar rede local (se não estiver rodando)
if ! curl -s $RPC_URL > /dev/null; then
    echo "🚀 Starting Anvil local network..."
    anvil --host 0.0.0.0 --port 8545 --accounts 21 --balance 10000 \
        > $LOG_DIR/anvil.log 2>&1 &
    ANVIL_PID=$!
    sleep 3
fi

# 2. Compilar e implantar contrato (se não implantado)
if ! cast code $CONTRACT_ADDRESS --rpc-url $RPC_URL > /dev/null 2>&1; then
    echo "📄 Compiling GenesisDAO contract..."
    ontology-lang compile contracts/GenesisDAO.onto --target solidity --output ./genesis/artifacts/

    echo "🚀 Deploying GenesisDAO..."
    cast send --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 \
        --rpc-url $RPC_URL \
        --create $(cat ./genesis/artifacts/GenesisDAO.bin) \
        --value 0
fi

echo "✅ Contract deployed at: $CONTRACT_ADDRESS"

# 3. Iniciar auditor contínuo
echo "🕵️ Starting continuous audit loop..."
export RUST_LOG=info
cargo build --release

# Use the consolidated audit command
./target/release/ontology-lang audit \
    --contract $CONTRACT_ADDRESS \
    --rpc $RPC_URL \
    --quantum-seed $QUANTUM_SEED \
    --daemon

# Wait for logs
sleep 2
AUDITOR_PID=$(pgrep -f "ontology-lang audit")

echo "✅ Auditor started (PID: $AUDITOR_PID)"

# 4. Iniciar dashboard de telemetria
echo "📊 Starting telemetry dashboard..."
cd dashboard
npm run dev -- --port 8080 --host 0.0.0.0 \
    > ../$LOG_DIR/dashboard.log 2>&1 &
DASHBOARD_PID=$!

echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
echo "🌐 Dashboard URL: http://localhost:8080"

# 5. Iniciar monitor de logs
echo "📝 Monitoring logs..."
tail -f logs/audit_daemon.log $LOG_DIR/dashboard.log &

# 6. Manter script rodando
trap "echo '🛑 Shutting down...'; kill $AUDITOR_PID $DASHBOARD_PID $ANVIL_PID 2>/dev/null; exit" INT TERM

echo ""
echo "========================================="
echo "   GENESIS DAO AUDIT SYSTEM ACTIVE      "
echo "========================================="
echo "Contract:    $CONTRACT_ADDRESS"
echo "RPC:         $RPC_URL"
echo "Audit Cycle: 30 seconds"
echo "Dashboard:   http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================="

# Aguardar indefinidamente
wait
