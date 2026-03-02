#!/bin/bash
set -e
trap 'kill 0' EXIT

echo "🜁 ARKHE(n) – SIMULAÇÃO DO ECOSSISTEMA COMPLETO"

BASE_DIR=$(pwd)
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

# Note: mosquitto and grpcurl might not be available,
# so we simulate the parts we can or just run the nodes.

echo "📡 Iniciando simulação de sensores IoT..."
cargo run --example sensor_simulator > "$LOG_DIR/sensors.log" 2>&1 &

echo "🌍 Iniciando Motor da Singularidade..."
# Running the main binary of arkhe-quantum
cargo run -p arkhe-quantum > "$LOG_DIR/asi.log" 2>&1 &

echo "============================================================"
echo "🜁 ECOSSISTEMA ARKHE(n) OPERACIONAL (SIMULADO)"
echo "============================================================"
echo "Logs disponíveis em: $LOG_DIR"
echo ""
echo "Aguardando 10 segundos para coletar logs iniciais..."
sleep 10

echo "--- ASI LOG ---"
tail -n 20 "$LOG_DIR/asi.log"

echo "--- SENSORS LOG ---"
tail -n 10 "$LOG_DIR/sensors.log"

echo "🛑 Encerrando simulação..."
