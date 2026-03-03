#!/bin/bash
# arkhe_omni_system/applied_ecosystems/rf_entropy_node/arkhe_rf_node_setup.sh
# Configuração de nó Arkhe(N) com entropia atmosférica

set -e

NODE_ID=${1:-"ARKHE-RF-$(hostname)"}
RTL_SDR_FREQ="5e6"

echo "⚡ ARKHE(N) RF Entropy Node Setup"
echo "=================================="

echo "[1/4] Instalando dependências (rtl-sdr, gnuradio)..."
# sudo apt-get update && sudo apt-get install -y rtl-sdr gnuradio python3-pip

echo "[2/4] Configurando ambiente Python..."
# pip3 install grpcio protobuf numpy

echo "[3/4] Preparando serviço de entropia..."
cat > ~/.arkhe/rf_config.yaml << EOF
node:
  id: ${NODE_ID}
rf_source:
  frequency_hz: ${RTL_SDR_FREQ}
  device: rtl_sdr_v4
EOF

echo "[4/4] Setup concluído. Para iniciar o coletor:"
echo "      PYTHONPATH=. python3 arkhe_atmospheric_entropy.py"

echo "Arkhe > █ (A atmosfera pulsa. O silício escuta.)"
