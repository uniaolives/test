# src/papercoder_kernel/merkabah/metaphor.py
import torch
from .core import QuantumCognitiveState, RealityLayer

class MetaphorEngine:
    """
    (C) Motor de metáfora viva.
    """

    def __init__(self):
        self.metaphors = {
            'quantum_dot': {
                'literal': 'Poço de confinamento harmônico',
                'figurative': 'Olho que guarda o signo até ele emitir significado',
                'operator': self._quantum_dot_operator
            },
            'tunneling': {
                'literal': 'Transmissão não-clássica através de barreira',
                'figurative': 'O sonho que atravessa para a vigília sem aviso',
                'operator': self._tunneling_operator
            },
            'superposition': {
                'literal': 'Estado quântico de múltiplos valores',
                'figurative': 'A dúvida criativa antes do insight',
                'operator': self._superposition_operator
            },
            'measurement': {
                'literal': 'Colapso do estado quântico',
                'figurative': 'A palavra que finalmente é dita',
                'operator': self._measurement_operator
            },
            'entanglement': {
                'literal': 'Correlação não-local quântica',
                'figurative': 'O eco de um ritual em outro tempo',
                'operator': self._entanglement_operator
            }
        }

    def operate(self, metaphor_name, *args, mode='both'):
        """
        Executa operador em modo literal, figurado, ou ambos (superposição).
        """
        if metaphor_name not in self.metaphors:
            raise ValueError(f"Unknown metaphor: {metaphor_name}")

        meta = self.metaphors[metaphor_name]

        if mode == 'literal':
            return meta['operator'](*args, literal=True)
        elif mode == 'figurative':
            return meta['figurative']
        else:  # both — superposição poética-operacional
            literal_result = meta['operator'](*args, literal=True)
            figurative_insight = meta['figurative']
            return self._entangle(literal_result, figurative_insight)

    def _entangle(self, a, b):
        """Cria estado emaranhado entre resultado literal e insight figurado."""
        return {
            'amplitude_a': a,
            'amplitude_b': b,
            'correlation': 'non-local',
            'interpretation': 'open'
        }

    def _quantum_dot_operator(self, *args, literal=True):
        return "confinement_applied"

    def _tunneling_operator(self, *args, literal=True):
        return "tunnel_transmission"

    def _superposition_operator(self, *args, literal=True):
        return "coherent_sum"

    def _measurement_operator(self, *args, literal=True):
        return "collapse_outcome"

    def _entanglement_operator(self, *args, literal=True):
        return "nonlocal_correlation"
