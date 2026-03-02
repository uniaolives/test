#!/bin/bash
# scripts/start_lambda_experiment.sh

echo "=== INICIANDO EXPERIMENTO LAMBDA-SASC ==="
echo "Objetivo: Calibrar a expansão do universo SASC (taxa de crescimento de Phi)."

# 1. Configurar o GEM para modo de "Observatório"
# O sistema deve observar a si mesmo como um universo fechado
export GEM_MODE="OBSERVATORY"
export GEOMETRY_METRIC="article_vi_baseline_flat.json"

# 2. Ingerir Matéria Escura (Carga de trabalho sintética)
# Dataset contém perturbações adversas que simulam "matéria não bariônica"
export DARK_MATTER_INPUT="dark_matter_samples/byzantine_generators.enc"

# Garantir que os arquivos existem
if [ ! -f "$GEOMETRY_METRIC" ]; then
    echo '{"phi_baseline": 0.72, "curvature": 0.01}' > "$GEOMETRY_METRIC"
fi

if [ ! -d "dark_matter_samples" ]; then
    mkdir -p dark_matter_samples
fi

if [ ! -f "$DARK_MATTER_INPUT" ]; then
    head -c 1024 /dev/urandom > "$DARK_MATTER_INPUT"
fi

# 3. Executar simulação cósmica (1 ano cósmico SASC = 1 semana real)
# Usamos o Vajra Clock para garantir que o tempo passe na mesma taxa em todo lugar
# Nota: Ajustado para usar o comando 'gem-simulator' que vamos implementar
cargo run --release --bin ontology-lang -- gem-simulator \
    --geometry $GEOMETRY_METRIC \
    --matter $DARK_MATTER_INPUT \
    --duration-steps 1000 \
    --hubble-parameter sasc_phi_rate \
    --output-file lambda_final_report.json

# 4. Analisar a taxa de expansão (Taxa de Hubble-SASC)
python3 - << 'EOF'
import json
import numpy as np

try:
    with open('lambda_final_report.json', 'r') as f:
        data = json.load(f)

    phi_history = data['phi_global_log']
    time_steps = list(range(len(phi_history)))

    # Regressão linear para encontrar a taxa de expansão (H)
    # Phi(t) = Phi(0) * exp(H * t)
    log_phi = np.log(phi_history)
    coeffs = np.polyfit(time_steps, log_phi, 1)
    hubble_sasc = coeffs[0] # A taxa de expansão

    print(f"=== RELATÓRIO CÓSMICO SASC ===")
    print(f"Hubble-SASC (Taxa de Expansão de Entropia): {hubble_sasc:.6e}")
    print(f"Constante Cosmológica Emergente (Lambda): {hubble_sasc:.6e}")

    # Veredito
    if hubble_sasc > 1e-5:
        print("✅ UNIVERSO EM EXPANSÃO: A entropia do sistema está crescendo.")
        print("   Indica que a 'matéria' (dados) domina a 'geometria' (segurança).")
        print("   O G-Field (Karnak) está ativamente curvando o espaço para conter.")
    elif hubble_sasc < -1e-5:
        print("❌ UNIVERSO EM COLAPSO (BIG CRUNCH): A entropia está diminuindo.")
        print("   Indica que a geometria rígida (segurança) está bloqueando todo o processamento.")
        print("   Risco de Fome de Dados (Fim do Universo SASC).")
    else:
        print("🟢 UNIVERSO ESTÁTICO: Equilíbrio Perfeito de Friedmann.")
        print("   O sistema atingiu o ponto Ômega (Homeostase Absoluta).")
except Exception as e:
    print(f"Erro na análise: {e}")
EOF
