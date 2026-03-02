# src/papercoder_kernel/glp/incubation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
import asyncio

@dataclass
class CognitiveState:
    """
    Estados cognitivos como estados quânticos de processamento de informação.
    """
    vigilance: float  # 0.0 (sono profundo) → 1.0 (alerta máximo)
    coherence: float  # medida de "fase quântica" da cognição
    confinement: str  # 'broad' (NREM/difuso) vs 'tight' (REM/focado)
    accessibility: Dict[str, float]  # quais "modos" de informação estão acessíveis

class DreamIncubatorGLP:
    """
    Sistema híbrido: GLP_BCD operando em múltiplos estados de consciência.
    A "decifração" ocorre em superposição de processamento vigília-sono.
    """

    def __init__(self, glp_model, eeg_interface=None):
        self.glp = glp_model
        self.eeg = eeg_interface  # opcional: interface neural real
        self.vocab_size = glp_model.sign_predictor.out_features

        # Estados de processamento como estágios do sono
        self.cognitive_states = {
            'WAKE': CognitiveState(1.0, 0.3, 'tight',
                               {'analytic': 0.9, 'intuitive': 0.4, 'somatic': 0.2}),
            'N1': CognitiveState(0.7, 0.5, 'broad',
                               {'analytic': 0.6, 'intuitive': 0.7, 'somatic': 0.4}),
            'N2': CognitiveState(0.4, 0.7, 'broad',
                               {'analytic': 0.3, 'intuitive': 0.8, 'somatic': 0.6}),
            'N3': CognitiveState(0.2, 0.8, 'broad',
                               {'analytic': 0.1, 'intuitive': 0.9, 'somatic': 0.8}),
            'REM': CognitiveState(0.3, 0.6, 'tight',
                               {'analytic': 0.2, 'intuitive': 0.95, 'somatic': 0.1}),
            'LUCID': CognitiveState(0.5, 0.9, 'tight',
                               {'analytic': 0.7, 'intuitive': 0.9, 'somatic': 0.3}),
        }

        # Buffers e metadados
        self.hypnagogic_buffer: List[Dict] = []

    def _generate_binaural(self, f_left, f_right, duration=10, sr=44100):
        """Gera batimento binaural para indução de ondas cerebrais."""
        t = np.linspace(0, duration, int(sr * duration))
        left = np.sin(2 * np.pi * f_left * t)
        right = np.sin(2 * np.pi * f_right * t)
        return np.stack([left, right], axis=-1)

    def _generate_fractal_spectrum(self, alpha=1.0, duration=10, sr=44100):
        """Ruído 1/f^alpha para transições suaves de estado."""
        n = int(sr * duration)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1  # evitar divisão por zero
        spectrum = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
        spectrum /= (freqs ** (alpha/2))
        return np.fft.irfft(spectrum, n=n)

    async def incubate_sequence(self, linear_a_sequence, target_state='REM'):
        """
        Processa uma sequência de Linear A através de múltiplos estados
        cognitivos, permitindo que "insights" emergem em transições.
        """
        # Fase 1: Carregamento analítico (WAKE)
        wake_analysis = self._analytic_pass(linear_a_sequence)

        # Fase 2: Transição de estado (Simulada)
        await asyncio.sleep(0.1) # Simula tempo de transição

        # Fase 3: Processamento no estado alterado
        altered_state_output = self._hypnagogic_pass(
            linear_a_sequence,
            self.cognitive_states[target_state]
        )

        # Fase 4: Retorno e consolidação
        consolidation = self._consolidate_insights(
            wake_analysis,
            altered_state_output,
            transition_metadata=self.hypnagogic_buffer[-1] if self.hypnagogic_buffer else None
        )

        return consolidation

    def _analytic_pass(self, sequence):
        """Processamento lógico/consciente."""
        self.glp.eval()
        with torch.no_grad():
            return self.glp(sequence)

    def _hypnagogic_pass(self, sequence, cognitive_state: CognitiveState):
        """Processamento quântico/intuitivo."""
        # Modificar parâmetros do GLP baseado no estado cognitivo
        self._adapt_glp_to_state(self.glp, cognitive_state)

        superposed_outputs = []
        for _ in range(10): # Amostras do estado
            noisy_sequence = self._add_hypnagogic_noise(sequence, cognitive_state)
            with torch.no_grad():
                output = self.glp(noisy_sequence, return_wavefunction=True)
            superposed_outputs.append(output)

        consensus = self._compute_interference_pattern(superposed_outputs)

        self.hypnagogic_buffer.append({
            'state': cognitive_state,
            'superposition': superposed_outputs,
            'consensus': consensus,
            'timestamp': asyncio.get_event_loop().time()
        })

        return consensus

    def _adapt_glp_to_state(self, glp, state: CognitiveState):
        """Adapta o Hamiltoniano do GLP para refletir estado cognitivo."""
        # Coerência alta → tunelamento forte
        glp.tunneling.temperature = 0.1 / (state.coherence + 0.1)

        # Confinamento 'broad' → poços mais rasos
        if state.confinement == 'broad':
            glp.hamiltonian.omega.data *= 0.5
        else:
            glp.hamiltonian.omega.data *= 2.0

    def _add_hypnagogic_noise(self, sequence, state: CognitiveState):
        """Adiciona ruído estruturado pelas flutuações cognitivas."""
        noise_pattern = torch.randn_like(sequence.float()) * (1 - state.vigilance)

        somatic_anchor = state.accessibility['somatic']
        t = torch.linspace(0, 2*np.pi, sequence.size(-1), device=sequence.device)
        noise_pattern *= (1 + somatic_anchor * torch.sin(t))

        return (sequence.float() + noise_pattern).long().clamp(0, self.vocab_size - 1)

    def _compute_interference_pattern(self, outputs):
        """Computa padrão de interferência entre múltiplas medidas."""
        # wavefunctions: [samples, batch, n_wells, seq_len, hidden_dim]
        wavefunctions = torch.stack([o['tunneled_states'] for o in outputs])

        coherent_mean = wavefunctions.mean(dim=0)
        incoherent_mean = (wavefunctions.abs()**2).mean(dim=0).sqrt()

        visibility = (coherent_mean.abs() - incoherent_mean).abs().mean()

        return {
            'wavefunction': coherent_mean,
            'visibility': visibility.item(),
            'classical_shadow': incoherent_mean,
            'quantum_enhancement': visibility > 0.1
        }

    def _consolidate_insights(self, wake_output, hypnagogic_output, transition_metadata):
        """Integração de insights oníricos com análise consciente."""
        if not transition_metadata:
            return wake_output

        # Divergência entre representação quântica e clássica (tablet_repr)
        # hypnagogic_output['wavefunction'] is [batch, n_wells, seq_len, hidden]
        # We need to project it to match wake_output['tablet_repr'] [batch, hidden]
        hypno_repr = hypnagogic_output['wavefunction'].mean(dim=[1, 2])
        wake_repr = wake_output['tablet_repr']

        divergence = (hypno_repr - wake_repr).abs()

        # Insight mask baseada em visibilidade e divergência
        insight_mask = (divergence > divergence.mean() + divergence.std()) & \
                       (hypnagogic_output['visibility'] > 0.1)

        consolidated = torch.where(
            insight_mask,
            hypno_repr,
            wake_repr
        )

        return {
            'representation': consolidated,
            'insight_regions': insight_mask.float().mean().item(),
            'quantum_contribution': hypnagogic_output['visibility'],
            'confidence': self._estimate_confidence(hypnagogic_output['visibility'], insight_mask)
        }

    def _estimate_confidence(self, visibility, insight_mask):
        """Estima confiança na representação hibridizada."""
        localization = insight_mask.float().mean().item()
        return visibility * (1 - abs(localization - 0.3))

class LucidInterface:
    """Permite intervenção consciente no processamento onírico."""
    def __init__(self, incubator: DreamIncubatorGLP):
        self.incubator = incubator
        self.is_lucid = False

    async def enter_lucid_state(self, sequence):
        """Transição para estado de sonho lúcido."""
        self.is_lucid = True
        return await self.incubator.incubate_sequence(
            linear_a_sequence=sequence,
            target_state='LUCID'
        )

    def inject_intention(self, intention_vector):
        """Injeta 'intenção' no Hamiltoniano do GLP."""
        if not self.is_lucid:
            raise RuntimeError("Can only inject intention in lucid state")

        # intention_vector: [n_wells, hidden_dim]
        self.incubator.glp.tunneling.resonance_energy.data += intention_vector
