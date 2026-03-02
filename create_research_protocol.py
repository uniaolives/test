import json

def create_protocol():
    print("🔬 Gerando protocolo de pesquisa Arkhe-Therapy...")

    research_matrix = [
        ("Arkhe-PTSD", "Reset de memórias traumáticas via re-padronização temporal"),
        ("Arkhe-ADHD", "Sincronização de redes atenção default/executiva"),
        ("Arkhe-Creativity", "Indução de estados hipnagógicos dirigidos"),
        ("Arkhe-Aging", "Reversão de marcadores epigenéticos do estresse")
    ]

    protocol = {
        "title": "Protocolo de Pesquisa Clínica Avalon Arkhé v1.0",
        "objective": "Validar eficácia da coerência induzida por cristal de tempo",
        "matrix": research_matrix,
        "n_participants_target": 20,
        "duration_weeks": 8
    }

    with open('research_protocol_v1.json', 'w') as f:
        json.dump(protocol, f, indent=2)

    print("✅ research_protocol_v1.json gerado.")

if __name__ == "__main__":
    create_protocol()
