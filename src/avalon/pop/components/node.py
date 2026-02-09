"""
Quantum-Classical Node (QCN) para o protocolo POP.
Implementação das seções 3.2 e 5.1 do whitepaper.
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
import json

from ..circuits.oracle import PersistentOrderOracle
from ..integration.qhttp import QHTTPClient, QuantumPacket

@dataclass
class SpectralCube:
    """Hipercubo espectral 4D (x, y, λ, t) simplificado"""
    data: np.ndarray  # Forma: (x, y, wavelengths, time_steps)
    coordinates: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class OperationalQCN:
    """
    Nó Quantum-Classical Node (QCN) operacional.
    Integra extração de features clássica com avaliação quântica.
    """

    def __init__(self, node_id: str, qhttp_gateway: str = "http://localhost:8080/qhttp"):
        self.node_id = node_id
        self.protocol_state = "SURVEILLANCE"
        self.oracle = PersistentOrderOracle()
        self.qhttp_client = QHTTPClient(node_id=node_id, qhttp_gateway=qhttp_gateway)
        self.detection_history = []

    async def initialize(self):
        await self.qhttp_client.connect()
        print(f"✅ Nó QCN {self.node_id} inicializado e conectado.")

    def extract_features(self, cube: SpectralCube) -> Dict[str, float]:
        """
        Extrai as três features fundamentais da Ordem Persistente.
        """
        data = cube.data

        # DNE: Desequilíbrio Não-Equilibrado Dinâmico (variância temporal)
        dne_score = np.std(data, axis=-1).mean()
        dne_norm = np.tanh(dne_score) # Normalizar para [0, 1]

        # SSO: Auto-Organização Espacial (correlação espacial entre fatias)
        sso_score = np.corrcoef(data[:, :, 0, 0].flatten(),
                                data[:, :, -1, 0].flatten())[0, 1]
        sso_norm = max(0, (sso_score + 1) / 2)

        # CDC: Acoplamento Domínio-Cruzado (correlação entre comprimentos de onda)
        # We look at how wavelengths correlate across all spatial and temporal samples
        wavelengths = data.shape[2]
        if wavelengths > 1:
            data_reshaped = np.moveaxis(data, 2, 0).reshape(wavelengths, -1)
            corr_matrix = np.corrcoef(data_reshaped)
            # Take mean of upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(wavelengths, k=1)
            cdc_score = np.mean(np.abs(corr_matrix[triu_indices]))
        else:
            cdc_score = 0.0
        cdc_norm = float(np.nan_to_num(cdc_score))

        return {
            'dne': float(dne_norm),
            'sso': float(sso_norm),
            'cdc': float(cdc_norm)
        }

    async def evaluate_cube(self, cube: SpectralCube) -> Dict:
        """
        Executa o pipeline completo: features -> oracle -> state update.
        """
        features = self.extract_features(cube)

        # Avaliação quântica
        result = self.oracle.simulate_detection(
            features['dne'],
            features['sso'],
            features['cdc']
        )

        # Atualizar estado do protocolo
        old_state = self.protocol_state
        self._update_protocol_state(result['detection_probability'])

        evaluation_result = {
            'node_id': self.node_id,
            'timestamp': cube.timestamp,
            'coordinates': cube.coordinates,
            'features': features,
            'detection_probability': result['detection_probability'],
            'is_life_detected': result['is_life_detected'],
            'protocol_state': self.protocol_state
        }

        self.detection_history.append(evaluation_result)

        if self.protocol_state != old_state:
            print(f"🔄 [{self.node_id}] Transição de estado: {old_state} -> {self.protocol_state}")

        return evaluation_result

    def _update_protocol_state(self, prob: float):
        if self.protocol_state == "SURVEILLANCE" and prob > 0.80:
            self.protocol_state = "CURIOSITY"
        elif self.protocol_state == "CURIOSITY" and prob > 0.95:
            self.protocol_state = "DISCOVERY"
        elif self.protocol_state == "CURIOSITY" and prob < 0.60:
            self.protocol_state = "SURVEILLANCE"
        elif self.protocol_state == "DISCOVERY" and prob < 0.90:
            self.protocol_state = "CURIOSITY"

    async def broadcast_alert(self, evaluation: Dict):
        """Propaga alerta via qhttp:// se a confiança for alta."""
        if evaluation['is_life_detected']:
            packet = QuantumPacket(
                sender_id=self.node_id,
                receiver_id="BROADCAST",
                classical_data=evaluation
            )
            await self.qhttp_client.send_quantum_packet(packet)

class MinimalQCN(OperationalQCN):
    """Versão simplificada para demonstração rápida."""
    pass
