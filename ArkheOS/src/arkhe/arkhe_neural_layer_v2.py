# arkhe_neural_layer_v2.py
import numpy as np
import time
import json
try:
    from .arkhe_error_handler import safe_operation, write_ledger
except ImportError:
    from arkhe_error_handler import safe_operation, write_ledger

class DeepNeuralLayer:
    """
    Camada Neural Profunda Arkhe (v2).
    Implementa uma arquitetura com múltiplas camadas ocultas, ReLU e regularização.
    """
    def __init__(self, input_size, hidden_sizes, output_size, node_id):
        self.node_id = node_id
        self.layers = []
        prev = input_size
        for h in hidden_sizes:
            # He initialization for ReLU
            self.layers.append({
                'weights': np.random.randn(prev, h) * np.sqrt(2. / prev),
                'bias': np.zeros(h) + 0.1
            })
            prev = h
        # Output layer (linear/regression for C prediction)
        self.layers.append({
            'weights': np.random.randn(prev, output_size) * 0.1,
            'bias': np.zeros(output_size)
        })
        self.activation_history = []

    def forward(self, x, training=False, dropout_rate=0.2):
        current_x = x
        for i, layer in enumerate(self.layers[:-1]):
            z = np.dot(current_x, layer['weights']) + layer['bias']
            current_x = np.maximum(0, z)  # ReLU

            if training:
                mask = np.random.binomial(1, 1 - dropout_rate, current_x.shape)
                current_x *= mask / (1 - dropout_rate)  # Inverted dropout

        # Last layer
        out = np.dot(current_x, self.layers[-1]['weights']) + self.layers[-1]['bias']
        if not training:
            self.activation_history.append(out.tolist())
        return out

    def forecast(self, current_state, steps=6):
        """
        Preve os próximos 'steps' estados baseado no estado atual.
        Nesta versão conceitual, usamos a rede de forma recursiva ou autoregressiva.
        """
        predictions = []
        state = current_state
        for _ in range(steps):
            state = self.forward(state)
            predictions.append(state)
        return np.array(predictions)

    @safe_operation
    def store_state(self):
        """Persiste os pesos de todas as camadas no ledger."""
        layers_data = []
        for l in self.layers:
            layers_data.append({
                'weights': l['weights'].tolist(),
                'bias': l['bias'].tolist()
            })

        entry = {
            'node_id': self.node_id,
            'layers': layers_data,
            'timestamp': time.time()
        }
        write_ledger(json.dumps(entry))
        return entry

    @classmethod
    def load_mock(cls, node_id):
        """Simula o carregamento de uma rede treinada."""
        # 11 sujeitos -> 11 preditos
        return cls(input_size=11, hidden_sizes=[16, 8], output_size=11, node_id=node_id)

if __name__ == "__main__":
    print("Testando Camada Neural Profunda (v2)...")
    net = DeepNeuralLayer(input_size=11, hidden_sizes=[16, 8], output_size=11, node_id='arkhe_neural_v2')

    # Simular dados de 11 sujeitos (coerência C)
    current_state = np.random.rand(11)
    print(f"Estado Atual (C): {current_state}")

    # Fazer previsão para as próximas 6 horas
    future = net.forecast(current_state, steps=6)
    print(f"Previsão (6 passos):\n{future}")

    # Verificar queda de coerência (threshold 0.85)
    anomalies = np.where(future < 0.85)
    if len(anomalies[0]) > 0:
        print(f"Alerta: Queda de coerência prevista detectada em {len(anomalies[0])} pontos.")

    net.store_state()
    print("Estado da v2 persistido.")
