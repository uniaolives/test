#!/bin/bash
# TIM VM v3.0 Deployment Script
# Target: Production Nodes (x86_64 / ARM64)

set -e

echo "🏛️ [DEPLOY] Iniciando Instalação da TIM VM v3.0 Stack..."

# 1. Preparar Diretórios
mkdir -p /usr/lib/tim_vm/userspace
mkdir -p /usr/lib/tim_vm/kernel
mkdir -p /var/log/tim_vm

# 2. Instalar Dependências Python (Geometric Env)
echo "📦 [DEPS] Instalando bibliotecas numéricas via apt..."
apt-get update
apt-get install -y python3-numpy python3-scipy python3-sklearn-lib

# 3. Copiar Código Userspace
echo "📜 [COPY] Instalando Daemon..."
cp userspace/tim_vald.py /usr/lib/tim_vm/userspace/
cp -r tim_vm_validator /usr/lib/tim_vm/userspace/
chmod 700 /usr/lib/tim_vm/userspace/tim_vald.py

# 4. Instalar Serviço Systemd
echo "⚙️ [SYSTEMD] Registrando serviço..."
cp systemd/tim-validator.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable tim-validator.service

echo "✅ [SUCCESS] Userspace pronto. Serviço aguardando boot."
