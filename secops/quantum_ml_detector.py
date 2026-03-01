import numpy as np
from sklearn import svm
import qiskit
from qiskit import QuantumCircuit, execute, Aer

class HybridQuantumMLDetector:
    """
    Detector híbrido: extrai features quânticas (fidelidade, entropia)
    e alimenta um classificador clássico (SVM, Isolation Forest).
    """
    def __init__(self):
        self.svm = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
        self.feature_buffer = []
        self.backend = Aer.get_backend('statevector_simulator')

    def extract_quantum_features(self, handover_data):
        """Codifica handover em circuito quântico e extrai features."""
        # handover_data esperado: dict com 'entropy_cost', 'half_life', 'type'
        qc = QuantumCircuit(2)
        # Codifica entropia e tipo em rotações
        qc.ry(handover_data.get('entropy_cost', 0.1) * np.pi, 0)
        qc.ry(handover_data.get('half_life', 1000.0) * 0.001 * np.pi, 1)
        if handover_data.get('type', 1) == 1:  # excitatory
            qc.cx(0, 1)
        job = execute(qc, self.backend)
        state = job.result().get_statevector()
        # Features: amplitudes reais e imaginárias
        return np.concatenate([state.real, state.imag])

    def train(self, handover_history):
        features = [self.extract_quantum_features(h) for h in handover_history]
        self.svm.fit(features)

    def predict(self, handover_data):
        feat = self.extract_quantum_features(handover_data)
        return self.svm.predict([feat])[0] == -1  # anomalia
