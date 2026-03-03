# arkhe_omni_system/applied_ecosystems/rf_entropy_node/arkhe_rf_entropy.py
# Bloco GNU Radio para conversão de entropia RF em seed Arkhe(N)

import numpy as np
import hashlib
import time

# Mocking GNU Radio if not available for import
try:
    from gnuradio import gr
except ImportError:
    # Minimal mock for gr.sync_block
    class gr:
        class sync_block:
            def __init__(self, name, in_sig, out_sig):
                self.name = name
                self.in_sig = in_sig
                self.out_sig = out_sig

class ArkheEntropyExtractor(gr.sync_block):
    """
    Extrai entropia de amostras RF e gera seeds para o sistema Arkhe(N).
    Implementa debiasing de von Neumann e condicionamento SHA3-256.
    """

    def __init__(self, sample_rate=2.56e6, coherence_window=0.1):
        gr.sync_block.__init__(
            self,
            name='Arkhe Entropy Extractor',
            in_sig=[np.float32],      # Magnitude do sinal
            out_sig=[np.uint8]        # Seed de 256 bits (32 bytes)
        )

        self.sample_rate = sample_rate
        self.coherence_window = coherence_window  # 100ms de integração

        # Buffer para acumulação de bits
        self.raw_bits = []
        self.seed_history = []
        self.phi_local = 0.0  # Coerência local do "clima" RF

    def work(self, input_items, output_items):
        in_sig = input_items[0]
        out = output_items[0]

        # 1. Detectar impulsos de raio (picos > 3σ)
        mean = np.mean(in_sig)
        std = np.std(in_sig)
        threshold = mean + 3 * std

        impulses = in_sig > threshold

        # 2. Extrair bits dos LSB das amostras de impulso
        for i, is_impulse in enumerate(impulses):
            if is_impulse:
                # Converter amplitude em bits (usando LSB do float32)
                sample_bytes = in_sig[i].tobytes()
                lsb_bits = [int(b) & 1 for b in sample_bytes]
                self.raw_bits.extend(lsb_bits)

        # 3. Debiasing de von Neumann (elimina viés)
        clean_bits = self._von_neumann_debias(self.raw_bits)

        # 4. Quando temos 256 bits limpos, gerar seed
        if len(clean_bits) >= 256:
            seed_bits = clean_bits[:256]
            seed_bytes = np.packbits(seed_bits)

            # 5. Condicionamento criptográfico
            hasher = hashlib.sha3_256()
            hasher.update(seed_bytes.tobytes())
            hasher.update(str(time.time()).encode())  # Timestamp
            final_seed = hasher.digest()

            # 6. Calcular Φ local (coerência do sinal RF)
            self.phi_local = self._calculate_rf_coherence(in_sig)

            # 7. Enviar para output
            # Ensure we don't overflow out
            bytes_to_copy = min(len(final_seed), len(out))
            out[:bytes_to_copy] = np.frombuffer(final_seed[:bytes_to_copy], dtype=np.uint8)

            self.seed_history.append({
                'seed': final_seed.hex(),
                'phi_rf': self.phi_local,
                'timestamp': time.time(),
                'impulse_count': int(np.sum(impulses))
            })

            # Limpar buffer para próxima rodada
            self.raw_bits = clean_bits[256:]

            return bytes_to_copy

        return 0  # Aguardando mais entropia

    def _von_neumann_debias(self, bits):
        """Elimina viés de sequência de bits."""
        result = []
        for i in range(0, len(bits) - 1, 2):
            if bits[i] != bits[i+1]:
                result.append(bits[i])  # 01 → 0, 10 → 1
        return result

    def _calculate_rf_coherence(self, signal):
        """
        Calcula 'coerência atmosférica' local.
        """
        if len(signal) == 0: return 0.0
        # Autocorrelação do sinal (simplificada)
        # We'll use a smaller slice for performance
        s = signal[:1000]
        if len(s) < 100: return 0.0
        autocorr = np.correlate(s, s, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Normalizar
        if autocorr[0] == 0: return 0.0
        autocorr = autocorr / autocorr[0]

        # Φ = área sob a curva de autocorrelação
        phi = np.sum(np.abs(autocorr[:100])) / 100.0

        return float(min(phi, 1.0))
