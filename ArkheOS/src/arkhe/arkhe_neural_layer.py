# arkhe_neural_layer.py
import numpy as np
import time
import json
try:
    from .arkhe_error_handler import safe_operation, write_ledger
except ImportError:
    from arkhe_error_handler import safe_operation, write_ledger

class NeuralLayer:
    def __init__(self, input_size, output_size, node_id):
        self.node_id = node_id
        # He initialization for ReLU
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros(output_size) + 0.1
        self.activation_history = []  # armazena ativações para auditoria

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        a = np.maximum(0, z)  # ReLU activation
        self.activation_history.append(a.tolist())
        return a

    @safe_operation
    def store_state(self):
        """Persiste pesos e bias no ledger do hipergrafo."""
        entry = {
            'node_id': self.node_id,
            'weights': self.weights.tolist(),
            'bias': self.bias.tolist(),
            'timestamp': time.time()
        }
        write_ledger(json.dumps(entry))
        return entry

if __name__ == "__main__":
    print("Testando Camada Neural Arkhe...")
    # Exemplo: camada que transforma dados de 11 sujeitos em representação de tríade
    layer = NeuralLayer(input_size=11, output_size=3, node_id='neural_triad')

    # Gerar dados de entrada aleatórios (simulando fMRI ou outros sensores)
    input_vector = np.random.rand(11)
    print(f"Entrada: {input_vector}")

    output = layer.forward(input_vector)
    print(f"Representação triádica (ReLU): {output}")

    print("Persistindo estado...")
    state = layer.store_state()
    print(f"Estado persistido para o nó: {state['node_id']}")
