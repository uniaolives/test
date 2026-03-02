# python/client.py
import requests
import json
import sys

# URL da API Elixir (O Cérebro)
ATLAS_URL = "http://localhost:4000"

def execute_task(action, target="github.com"):
    """Envia uma tarefa para o Cérebro da AGI"""

    payload = {
        "action": action,
        "target": target
    }

    print(f"🧠 [PYTHON] Iniciando intenção: {action}")

    try:
        # 1. Envia para Elixir (que consulta Rust e roteia)
        response = requests.post(f"{ATLAS_URL}/task", json=payload)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ [PYTHON] Tarefa aceita: {data['status']}")
            print(f"   Processado por: {data['processed_by']}")
        else:
            print(f"🛡️ [PYTHON] Bloqueado: {response.json()['error']}")

    except Exception as e:
        print(f"❌ [PYTHON] Erro de conexão: {e}")

def check_status():
    """Verifica o estado global (via Rust NIF)"""
    try:
        response = requests.get(f"{ATLAS_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"🧠 [ATLAS STATE]")
            print(f"   Intenção: {data['intent']}")
            print(f"   Tom Emocional: {data['emotion']:.2f}")
        else:
            print("⚠️ [PYTHON] Cérebro offline.")
    except Exception as e:
        print(f"❌ [PYTHON] Erro de conexão: {e}")

if __name__ == "__main__":
    # Exemplo de uso
    if len(sys.argv) < 2:
        print("Uso: python client.py <buscar|status> [url]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "buscar":
        target = sys.argv[2] if len(sys.argv) > 2 else "github.com/trending"
        execute_task("search", target)
    elif command == "status":
        check_status()
