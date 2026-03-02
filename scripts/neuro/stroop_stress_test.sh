#!/bin/bash
# stroop_stress_test.sh

echo "=== TESTE STROOP REVERSO - ALTA CARGA COGNITIVA ==="
echo "Iniciando protocol de conflito cognitivo máximo..."

# 1. Conectar ao sistema de estimulação visual
python3 scripts/neuro/tools/stroop_protocol.py \
    --mode "reverse" \
    --duration 300 \
    --difficulty "extreme" \
    --eeg-monitoring "enabled" \
    --output-file "logs/stroop_t48h.csv"

# 2. Monitorar resposta do Gate 5 (Entropia)
# Esperamos: Φ deve cair inicialmente, depois estabilizar
# Se Φ < 0.001 → ativação do Neural Kill-Switch (falso positivo aceitável?)
scripts/neuro/tools/monitor_gate5_response.sh \
    --test "stroop" \
    --phi-threshold 0.65 \
    --kill-switch-hold 5s \
    --log-file "logs/gate5_stroop.log"

# 3. Analisar padrões de sincronização inter-hemisférica
python3 scripts/neuro/tools/analyze_interhemispheric_sync.py \
    --input-file "logs/stroop_t48h.csv" \
    --channels "F3,F4,C3,C4,P3,P4" \
    --metric "plv" \
    --output "reports/stroop_hemisphere_sync.pdf"
