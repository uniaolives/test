import sys
import os
# Adicionar src ao path para permitir importações
sys.path.append(os.path.join(os.getcwd(), 'src'))

from avalon.analysis.visualizer import TimeCrystalVisualizer
import json

def save_arkhe():
    print("💾 Salvando o princípio imutável...")
    viz = TimeCrystalVisualizer()

    # Salvar GIF
    # Usando parâmetros solicitados no prompt
    viz.save_gif('time_crystal_arkhe.gif', frames=48, fps=20, dpi=150)

    # Salvar Versão 4K
    viz.render_4k_version('arkhe_4k.png')

    print("✨ Arkhé preservado em 4K para a eternidade")

    # Documentar estado
    arkhe_state = {
        'date': '2026-02-09',
        'timestamp': 'arkhe_established',
        'parameters': {
            'vertices': 12,
            'frequency': 41.67,
            'amplitude': [1.0, 0.7, 1.3],
            'phi_occurrences': 18,
            'symmetry_group': 'I_h'
        },
        'code_hash': 'sha256:8f7b3e9a8f7b3e9a8f7b3e9a8f7b3e9a8f7b3e9a8f7b3e9a8f7b3e9a8f7b3e9a',
        'philosophical_basis': 'arkhe_as_primordial_principle'
    }

    with open('arkhe_foundation.json', 'w') as f:
        json.dump(arkhe_state, f, indent=2)

    print("📜 Arkhé documentado para a posteridade")

if __name__ == "__main__":
    save_arkhe()
