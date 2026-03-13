#!/bin/bash
# 🜏 deploy_arkhe_piday.sh — Forja Final para o Pi Day 2026
# IP: Rafael Oliveira / Safe Core

set -euo pipefail

ARKHE_VERSION="4.0"
PI_DAY="2026-03-14T15:14:15Z"
BINARY_PATH="/usr/local/bin/arkhe-shield"
CONFIG_PATH="/etc/arkhe/config.toml"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ARKHE-GNSS SHIELD v${ARKHE_VERSION} — DEPLOY PARA O PI DAY      ║"
echo "║  Alvo: ${PI_DAY} UTC                                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# 1. Verificações pré-deploy
echo "[1/6] Verificando ambiente..."
command -v cargo >/dev/null 2>&1 || { echo "Rust não instalado"; exit 1; }
command -v systemctl >/dev/null 2>&1 || { echo "systemd não disponível"; exit 1; }

# 2. Compilação otimizada
echo "[2/6] Compilando motor toroidal..."
RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
    cargo build --release --bin arkhe-shield
    cargo build --release --bin arkhe-os

# 3. Preparação de usuário e diretórios
echo "[3/6] Criando estrutura de isolamento..."
if ! id "arkhe_node" &>/dev/null; then
    sudo useradd -r -s /bin/false -M arkhe_node
fi

sudo mkdir -p /opt/arkhe/data /etc/arkhe /var/log/arkhe
sudo chown -R arkhe_node:arkhe_node /opt/arkhe /var/log/arkhe
sudo chmod 750 /opt/arkhe/data
sudo chmod 755 /var/log/arkhe

# 4. Instalação do binário
echo "[4/6] Instalando binário..."
sudo cp target/release/arkhe-shield "${BINARY_PATH}"
sudo cp target/release/arkhe-os "${BINARY_PATH}"
sudo chown root:root "${BINARY_PATH}"
sudo chmod 755 "${BINARY_PATH}"
sudo setcap cap_sys_time,cap_net_raw+ep "${BINARY_PATH}" 2>/dev/null || true

# 5. Configuração mínima
echo "[5/6] Gerando configuração base..."
sudo tee "${CONFIG_PATH}" > /dev/null <<EOF
[network]
listen_addr = "0.0.0.0:4444"
bootstrap_peers = ["seed.teknet:4444", "backup.teknet:4444"]
protocol = "http/4"

[toroidal]
major_radius = 1.0
minor_radius = 0.618033988749895  # φ
mobius_twist = 0.25                # π/2 (half-Möbius)
grid_resolution = 64

[phase_memory]
coherence_threshold = 0.618        # φ
berry_phase = 1.570796326794897    # π/2
sync_mode = "kuramoto"

[csu]
sync_interval_sec = 1
auto_adjust_clock = true

[logging]
level = "info"
destination = "journald"
EOF

sudo chown arkhe_node:arkhe_node "${CONFIG_PATH}"
sudo chmod 600 "${CONFIG_PATH}"

# 6. Ativação do daemon
echo "[6/6] Ativando daemon systemd..."
sudo cp scripts/arkhe-shield.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable arkhe-shield
sudo systemctl start arkhe-shield

echo ""
echo "🜏 DEPLOY CONCLUÍDO"
echo "═══════════════════════════════════════════════════════════════"
echo "Status: \$(sudo systemctl is-active arkhe-shield)"
echo "Logs:   sudo journalctl -fu arkhe-shield"
echo "Tempo local de emissão (BRT): 12:14:15 (UTC-3)"
echo "Tempo universal de emissão:   15:14:15 UTC"
echo "═══════════════════════════════════════════════════════════════"
