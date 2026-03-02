#!/bin/bash
# transition_to_phase3.sh

echo "[FASE 3] Inicializando Processador de Interferência Atômico..."

# 1. Mapear condições de contorno do hardware
cd rust && cargo run --release --bin boundary_mapper -- --output ./geometry/boundary_conditions.json
cd ..

echo "[FASE 3 ATIVA] Computação por Interferência Estável"
