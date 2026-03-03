# sim_rj_geodesic.py
"""
Simulação de Trajetórias Reais no Rio de Janeiro.
Mapeia a geodésica urbana do dispositivo do Arquiteto.
"""

from datetime import datetime, timedelta
import random
from arkhe.arkhe_rfid import VirtualDeviceNode, RFIDHypergraph

def run_rj_simulation():
    # 1. Inicializar Nó e Hipergrafo
    coords_copa = (-22.9711, -43.1825)
    device = VirtualDeviceNode("Γ_DEVICE_RAFAEL_20260215", "Smartphone", coords_copa)
    hg = RFIDHypergraph()
    hg.add_tag(device)

    # 2. Definir Pontos da Geodésica
    points = [
        ("10:00", "Copacabana", (-22.9711, -43.1825)),
        ("11:30", "Ipanema", (-22.9847, -43.2031)),
        ("13:00", "Lagoa", (-22.9672, -43.2105)),
        ("15:00", "Jardim Botânico", (-22.9660, -43.2267)),
        ("17:00", "Corcovado", (-22.9519, -43.2105)),
        ("19:00", "Lapa", (-22.9133, -43.1806)),
    ]

    print("--- Início da Geodésica Urbana (RJ) ---")
    current_time = datetime(2026, 2, 15, 10, 0)

    for time_str, loc, coords in points:
        # Registrar handover
        hg.register_reading(device.tag_id, f"READER_{loc}", loc, current_time)
        device.coords = coords
        print(f"[{time_str}] Handover em {loc} | C={device.coherence_history[-1]['C']:.2f}")

        # Simular anomalia específica (Corcovado)
        if loc == "Corcovado":
            device.simulate_anomaly("Leitura Perdida")
            print(f"  ⚠️ Anomalia detectada: Leitura Perdida (Sinal Fraco)")

        current_time += timedelta(minutes=90)

    return device, hg

if __name__ == "__main__":
    run_rj_simulation()
