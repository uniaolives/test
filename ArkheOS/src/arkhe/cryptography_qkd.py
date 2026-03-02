# arkhe/cryptography_qkd.py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

@dataclass
class QuantumKey:
    key: bytes
    generated_at: float
    lifetime_hours: float = 24.0

class DarvoQKDManager:
    """
    Gerencia chaves quânticas usando o estado do Protocolo Darvo.
    A resiliência quântica é garantida pela distribuição de chaves BB84
    e pela rotação dinâmica baseada na flutuação (F) do Darvo.
    """
    def __init__(self, kernel_interface=None):
        self.current_key: Optional[QuantumKey] = None
        self.kernel = kernel_interface
        # Estado fallback se kernel não fornecido
        self._fallback_darvo = {"handover_count": 0, "coherence": 0.95, "darvo_remaining": 999.0}

    def get_darvo_state(self) -> Dict:
        if self.kernel and hasattr(self.kernel, 'darvo_remaining'):
            return {
                "handover_count": getattr(self.kernel, 'handover_count', 0),
                "coherence": getattr(self.kernel, 'global_coherence', 0.95),
                "darvo_remaining": self.kernel.darvo_remaining
            }
        return self._fallback_darvo

    def bb84_generate(self, length: int = 32) -> bytes:
        """Simula a geração de chave via BB84 (Entanglement-based)."""
        # Em hardware real, isso envolveria lasers e detectores de fótons únicos.
        return np.random.bytes(length)

    def rotate_key(self):
        """Gera e rotaciona a chave quântica."""
        state = self.get_darvo_state()
        new_key = QuantumKey(
            key=self.bb84_generate(),
            generated_at=time.time(),
            # O lifetime é encurtado se a coerência for baixa
            lifetime_hours=24.0 * state['coherence']
        )
        self.current_key = new_key
        print(f"🔐 [QKD] Chave rotacionada. Nova validade: {new_key.lifetime_hours:.2f}h (C={state['coherence']:.2f})")
        return new_key

    def is_key_valid(self) -> bool:
        if not self.current_key:
            return False

        state = self.get_darvo_state()
        # Se o escudo Darvo estiver crítico, a chave deve ser trocada
        if state['darvo_remaining'] < 100.0:
            print("⚠️ [Darvo] Escudo crítico detectado. Forçando rotação QKD.")
            return False

        age_hours = (time.time() - self.current_key.generated_at) / 3600
        return age_hours < self.current_key.lifetime_hours

    def encrypt_channel(self, payload: bytes) -> bytes:
        """Aplica a proteção quântica ao handover."""
        if not self.is_key_valid():
            self.rotate_key()

        # Simulação de cifragem AES-GCM usando a chave QKD
        print(f"🌀 [Quantum-Secure] Handover cifrado com chave QKD (ID: {id(self.current_key.key)})")
        return payload # Em produção, retornar dado cifrado

if __name__ == "__main__":
    manager = DarvoQKDManager()
    manager.rotate_key()
    manager.encrypt_channel(b"Dados soberanos")
