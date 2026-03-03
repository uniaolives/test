#!/bin/bash
# arkhe-axos-instaweb/deploy-arkhe.sh
set -e

echo "🜁 ARKHE-AXOS-INSTAWEB v1.0.0 DEPLOY"

# 1. Build
echo "[1/3] Building binaries..."
cargo build --release

# 2. Docker
echo "[2/3] Building docker images..."
docker-compose -f docker/docker-compose.yml build

# 3. Complete
echo "🜁 Deploy artifacts prepared."
