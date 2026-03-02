# arkhe_error_handler.py
import logging
import time
from functools import wraps
import os

# Ensure the log directory exists if we decide to put it in a specific folder,
# but for now we'll keep it simple as per the block.
LOG_FILE = 'arkhe_core.log'

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def safe_operation(func):
    """Decorator para envolver operações críticas em try-catch."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Erro em {func.__name__}: {str(e)}")
            # Em produção, poderia acionar protocolo de recuperação
            raise  # ou retornar um fallback
    return wrapper

# Exemplo de uso em operações de rede
@safe_operation
def fetch_node_data(node_id):
    # Simula requisição a um nó remoto
    if node_id == 42:
        raise ConnectionError("Nó 42 não respondeu ao handshake")
    return {"C": 0.95, "F": 0.05}

@safe_operation
def write_ledger(entry):
    # Simula escrita em arquivo
    # Using a relative path or a temporary one for safety in this environment
    with open("arkhe_ledger_test.txt", "a") as f:
        f.write(entry + "\n")

if __name__ == "__main__":
    print("Testando Arkhe Error Handler...")
    try:
        data = fetch_node_data(1)
        print(f"Dados obtidos: {data}")
        write_ledger(str(data))
        print("Escrita no ledger concluída.")

        print("Simulando erro no nó 42...")
        fetch_node_data(42)
    except Exception:
        print("Erro capturado e logado conforme esperado.")
