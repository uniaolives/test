from .safe_core import SafeCore, QuantumState
import numpy as np
import hashlib
import time
from typing import Dict, Any, List

class QuantumHandoverProtocol:
    """
    Protocolo de handover bidirecional quântico-clássico.
    Permite transferir estado entre nós ou congelar para backup.
    """

    def __init__(self):
        self.handover_history: List[Dict[str, Any]] = []

    def freeze_quantum_state(self, safe_core: SafeCore) -> Dict[str, Any]:
        """
        Congela o estado quântico para extração segura.
        Retorna um dicionário com o estado serializável.
        """
        # Aplicar dynamical decoupling (simulado com identidade para este protótipo)
        safe_core.apply_gate(np.eye(2**safe_core.n_qubits), list(range(safe_core.n_qubits)))

        # Extrair estado
        state = safe_core.extract_state()

        # Serializar para envio
        frozen = {
            'amplitudes': state.amplitudes.tolist(),
            'basis_labels': state.basis_labels,
            'coherence': safe_core.coherence,
            'phi': safe_core.phi,
            'timestamp': time.time_ns(),
            'hash': self._hash_state(state.amplitudes)
        }
        return frozen

    def transfer_to_classical(self, frozen: Dict[str, Any]) -> np.ndarray:
        """
        Converte estado quântico congelado em uma política clássica.
        Simula tomografia e aproximação por rede neural.
        """
        # Aqui faria tomografia quântica e treinamento de rede neural.
        # Por simplicidade, retorna a matriz densidade como array.
        amplitudes = np.array(frozen['amplitudes'], dtype=np.complex128)
        density = np.outer(amplitudes, amplitudes.conj())
        return density

    def resume_quantum(self, safe_core: SafeCore, frozen: Dict[str, Any]):
        """
        Restaura estado quântico a partir de checkpoint.
        """
        # Verificar integridade
        if not self._verify_hash(frozen):
            raise ValueError("Quantum state integrity check failed")

        # Recarregar estado
        amplitudes = np.array(frozen['amplitudes'], dtype=np.complex128)
        state = QuantumState(
            amplitudes=amplitudes,
            basis_labels=frozen['basis_labels']
        )
        safe_core.load_state(state)

        # Registrar
        self.handover_history.append({
            'event': 'resume',
            'timestamp': time.time_ns(),
            'hash': frozen['hash']
        })

    def _hash_state(self, amplitudes: np.ndarray) -> str:
        data = amplitudes.tobytes()
        return hashlib.sha256(data).encode().hexdigest()[:16] if hasattr(hashlib.sha256(data), 'encode') else hashlib.sha256(data).hexdigest()[:16]

    def _verify_hash(self, frozen: Dict[str, Any]) -> bool:
        computed = self._hash_state(np.array(frozen['amplitudes'], dtype=np.complex128))
        return computed == frozen['hash']

    # Correction for _hash_state to be more robust
    def _hash_state(self, amplitudes: np.ndarray) -> str:
        data = amplitudes.tobytes()
        return hashlib.sha256(data).hexdigest()[:16]
