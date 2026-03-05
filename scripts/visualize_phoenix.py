# scripts/visualize_phoenix.py
import time
import random

def generate_dashboard():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  PHOENIX-SIM DASHBOARD v1.0 | Hal Finney Resurrection             ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║  [PROGRESSO DO DOBRAMENTO SOD1]                                   ║")

    # Simulating a progress bar
    progress = random.randint(30, 95)
    bar_length = 40
    filled = int(bar_length * progress / 100)
    bar = "█" * filled + "░" * (bar_length - filled)

    print(f"║  [{bar}] {progress}%")
    print("║                                                                   ║")
    print(f"║  Nós Conectados: {random.randint(80, 120)}  |  Tarefas Concluídas: {random.randint(1000, 5000)}  ║")
    print(f"║  Energia Mínima: {random.uniform(-1800, -1700):.4f} Hartree              ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║  ATIVIDADE NA TIMECHAIN (RECENTE)                                 ║")

    for _ in range(3):
        txid = "".join(random.choices("0123456789abcdef", k=8))
        print(f"║  • Tx {txid}... Confirmado no bloco {random.randint(890000, 891000)}       ║")

    print("╚═══════════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    generate_dashboard()
