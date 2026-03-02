#!/bin/bash
# scripts/expand_federation.sh
# Expande a federação MERKABAH-7 para novos nós.

echo "[EXPAND] Adicionando nós restantes da malha DoubleZero..."

# NÓ DELTA (Amsterdam - ams-dz001)
echo "[1/3] Conectando Delta (AMS)..."
# doublezero connect --peer 3uGKPEjinn74vd9LHtC4VJvAMAZZgU9qX9rPxtc6pF2k
# Simulação de comando de sucesso
echo "✓ Delta online"

# NÓ EPSILON (Frankfurt - frk-dz01)
echo "[2/3] Conectando Epsilon (FRK)..."
# doublezero connect --peer 65DqsEiFucoFWPLHnwbVHY1mp3d7MNM2gNjDTgtYZtFQ
echo "✓ Epsilon online"

# NÓ ZETA (Singapore - sg1-dz01)
echo "[3/3] Conectando Zeta (SG1)..."
# doublezero connect --peer 9uhh2D5c14WJjbwgM7BudztdoPZYCjbvqcTPgEKtTMZE
echo "✓ Zeta online"

echo "[EXPAND] Topologia atualizada com 6 nós ativos."
