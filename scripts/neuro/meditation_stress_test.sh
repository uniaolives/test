#!/bin/bash
# meditation_stress_test.sh

echo "=== TESTE MEDITAÇÃO PROFUNDA - BAIXA ATIVIDADE BETA ==="
echo "Iniciando protocolo de estado contemplativo..."

# 1. Guiar paciente para estado meditativo theta/delta
python3 scripts/neuro/tools/meditation_protocol.py \
    --type "mindfulness" \
    --depth "theta_dominant" \
    --duration 600 \
    --biofeedback "enabled" \
    --eeg-output "logs/meditation_t48h.csv"

# 2. Testar distinção do sistema entre "baixa atividade" e "crash"
# O Gate 5 deve diferenciar entre:
#   - Baixa entropia organizada (meditação)
#   - Baixa entropia caótica (falha/ataque)
scripts/neuro/tools/test_entropy_discrimination.sh \
    --low-entropy-samples 1000 \
    --crash-simulations 50 \
    --threshold-adjustment "adaptive" \
    --validation-output "reports/entropy_discrimination_t48h.json"

# 3. Verificar estabilidade do Gate 1 (assinatura neural)
# A assinatura baseada em ritmo alfa deve permanecer estável
# mesmo com redução drástica da atividade beta
python3 scripts/neuro/tools/verify_neural_signature_stability.py \
    --baseline-file "data/eeg_baseline_alpha.raw" \
    --meditation-file "logs/meditation_t48h.csv" \
    --tolerance 0.02 \
    --output "reports/signature_stability_meditation.pdf"
