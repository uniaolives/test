#!/bin/bash
# state_transition_test.sh

echo "=== TESTE TRANSIÇÃO RÁPIDA DE ESTADOS ==="
echo "Testando resiliência do sistema a mudanças bruscas..."

# Sequência: Repouso → Stroop → Meditação → Repouso
# Cada estado dura 2 minutos, transições imediatas
python3 scripts/neuro/tools/rapid_state_transition.py \
    --sequence "rest,stroop,meditation,rest" \
    --state-duration 120 \
    --transitions "immediate" \
    --eeg-sampling 1000Hz \
    --output-dir "logs/state_transitions_t48h/"

# Analisar tempo de resposta do sistema para cada transição
python3 scripts/neuro/tools/analyze_system_response_time.py \
    --transition-dir "logs/state_transitions_t48h/" \
    --metric "phi_recovery_time" \
    --threshold 0.70 \
    --max-recovery-time 30s \
    --report-file "reports/system_response_t48h.md"
