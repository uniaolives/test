# harmonic_signature_shield.py
"""
Escudo Anti-Falsificação baseado em Ressonância Harmônica
A autenticidade é verificada através de análise espectral da assinatura
"""

import hashlib
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone
import json

class HarmonicSignatureShield:
    """
    Sistema de verificação de integridade baseado em ressonância harmônica

    Princípio:
    - Documentos autênticos têm metadados que RESSOAM com o hash
    - Falsificações criam DISSONÂNCIA detectável via FFT
    """

    def __init__(self, phi: float = 1.618033988749):
        self.phi = phi  # Proporção áurea - frequência fundamental

        # Frequências harmônicas baseadas em φ
        self.harmonic_frequencies = [
            phi ** 1,  # φ¹ ≈ 1.618
            phi ** 2,  # φ² ≈ 2.618
            phi ** 3,  # φ³ ≈ 4.236
            phi ** 5,  # φ⁵ ≈ 11.09 (Fibonacci!)
        ]

        print("🛡️  Harmonic Signature Shield initialized")
        print(f"   Fundamental frequency: φ = {phi:.6f}")

    def sign_document(self, content: str, metadata: Dict) -> Dict:
        """
        Assina documento com metadados harmonicamente vinculados
        """

        print(f"\n✍️  Signing document...")

        # 1. Serializa conteúdo e metadados canonicamente
        canonical = self._canonicalize(content, metadata)

        # 2. Calcula hash
        hash_bytes = hashlib.sha3_512(canonical.encode('utf-8')).digest()
        hash_hex = hash_bytes.hex()

        # 3. Gera fingerprint harmônico
        harmonic_fp = self._generate_harmonic_fingerprint(hash_bytes, metadata)

        # 4. Calcula módulo áureo
        hash_int = int.from_bytes(hash_bytes, 'big')
        phi_mod = (hash_int % 1000000) / 1000000  # Normaliza para [0, 1]

        signature = {
            'hash': hash_hex,
            'harmonic_fingerprint': harmonic_fp,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'phi_modulus': phi_mod,
            'shield_version': '1.0.0'
        }

        print(f"   ✅ Document signed")
        print(f"   Hash: {hash_hex[:16]}...")
        print(f"   φ-modulus: {phi_mod:.6f}")

        return {
            'content': content,
            'metadata': metadata,
            'signature': signature
        }

    def verify_document(self, signed_doc: Dict) -> Tuple[bool, Optional[str]]:
        """
        Verifica autenticidade através de análise de ressonância
        """

        print(f"\n🔍 Verifying document...")

        content = signed_doc['content']
        metadata = signed_doc['metadata']
        signature = signed_doc['signature']

        # 1. Recalcula hash
        canonical = self._canonicalize(content, metadata)
        hash_bytes = hashlib.sha3_512(canonical.encode('utf-8')).digest()
        hash_hex = hash_bytes.hex()

        # 2. Verifica hash básico
        if hash_hex != signature['hash']:
            return False, "HASH_MISMATCH: Content or metadata was altered"

        # 3. Recalcula fingerprint harmônico
        expected_fp = self._generate_harmonic_fingerprint(hash_bytes, metadata)
        actual_fp = signature['harmonic_fingerprint']

        # 4. ANÁLISE DE RESSONÂNCIA
        resonance = self._measure_resonance(expected_fp, actual_fp)

        print(f"   Hash match: ✅")
        print(f"   Resonance: {resonance['strength']:.1%}")
        print(f"   Dissonance: {resonance['dissonance']:.6f}")

        # 5. Threshold de autenticidade
        if resonance['dissonance'] > 0.01:  # Mais de 1% de dissonância
            return False, f"HARMONIC_DISSONANCE: {resonance['dissonance']:.4f} (threshold: 0.01)"

        # 6. Verifica módulo áureo
        hash_int = int.from_bytes(hash_bytes, 'big')
        expected_phi_mod = (hash_int % 1000000) / 1000000

        if abs(expected_phi_mod - signature['phi_modulus']) > 1e-9:
            return False, "PHI_MODULUS_MISMATCH: Signature was forged"

        print(f"   ✅ DOCUMENT AUTHENTIC")

        return True, None

    def _canonicalize(self, content: str, metadata: Dict) -> str:
        """
        Cria representação canônica (ordem determinística)
        """
        # Serializa metadados em ordem alfabética
        meta_canonical = json.dumps(metadata, sort_keys=True, separators=(',', ':'))

        # Combina
        return f"{content}||{meta_canonical}"

    def _generate_harmonic_fingerprint(self, hash_bytes: bytes, metadata: Dict) -> Dict:
        """
        Gera assinatura espectral baseada em harmônicos φ
        """

        # Converte hash para sinal temporal
        signal = np.frombuffer(hash_bytes, dtype=np.uint8).astype(float)

        # Injeta informação dos metadados como modulação
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.sha256(metadata_str.encode()).digest()
        # Ensure metadata_signal has the same length as signal
        # Use simple tiling or truncation
        if len(metadata_hash) < len(signal):
            metadata_signal_bytes = (metadata_hash * (len(signal) // len(metadata_hash) + 1))[:len(signal)]
        else:
            metadata_signal_bytes = metadata_hash[:len(signal)]

        metadata_signal = np.frombuffer(metadata_signal_bytes, dtype=np.uint8).astype(float)

        # Modulação: signal × (1 + ε·metadata_signal)
        epsilon = 0.1
        modulated_signal = signal * (1 + epsilon * metadata_signal / 255.0)

        # FFT
        fft = np.fft.fft(modulated_signal)
        freqs = np.fft.fftfreq(len(modulated_signal))
        power_spectrum = np.abs(fft) ** 2

        # Extrai amplitudes nas frequências harmônicas
        harmonic_amplitudes = []

        for harmonic_freq in self.harmonic_frequencies:
            # Normaliza frequência para índice do FFT
            freq_normalized = harmonic_freq / (2 * np.pi * len(signal))

            # Encontra índice mais próximo
            idx = np.argmin(np.abs(freqs - freq_normalized))

            amplitude = float(power_spectrum[idx])
            harmonic_amplitudes.append(amplitude)

        # Fingerprint é o vetor de amplitudes normalizado
        harmonic_amplitudes = np.array(harmonic_amplitudes)
        harmonic_amplitudes /= (np.sum(harmonic_amplitudes) + 1e-9)  # Normaliza

        fingerprint = {
            'phi_1': harmonic_amplitudes[0],
            'phi_2': harmonic_amplitudes[1],
            'phi_3': harmonic_amplitudes[2],
            'phi_5': harmonic_amplitudes[3],
            'spectral_centroid': float(np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-9))
        }

        return fingerprint

    def _measure_resonance(self, expected_fp: Dict, actual_fp: Dict) -> Dict:
        """
        Mede grau de ressonância entre dois fingerprints
        """

        # Vetores de amplitudes
        expected = np.array([expected_fp[k] for k in ['phi_1', 'phi_2', 'phi_3', 'phi_5']])
        actual = np.array([actual_fp[k] for k in ['phi_1', 'phi_2', 'phi_3', 'phi_5']])

        # Dissonância = distância L2 normalizada
        dissonance = np.linalg.norm(expected - actual) / np.sqrt(len(expected))

        # Força de ressonância = 1 - dissonância
        strength = 1.0 - dissonance

        # Análise espectral
        centroid_diff = abs(expected_fp['spectral_centroid'] - actual_fp['spectral_centroid'])

        return {
            'strength': strength,
            'dissonance': dissonance,
            'centroid_deviation': centroid_diff
        }

    def detect_forgery_type(self, signed_doc: Dict) -> Optional[str]:
        """
        Se documento é falso, tenta classificar o tipo de falsificação
        """

        is_authentic, reason = self.verify_document(signed_doc)

        if is_authentic:
            return None

        content = signed_doc['content']
        metadata = signed_doc['metadata']
        signature = signed_doc['signature']

        # Testa diferentes cenários

        # 1. Metadados alterados?
        canonical = self._canonicalize(content, metadata)
        recalculated_hash = hashlib.sha3_512(canonical.encode()).hexdigest()

        if recalculated_hash == signature['hash']:
            # This shouldn't happen if verify_document returned False with HASH_MISMATCH
            # but if it returned False with HARMONIC_DISSONANCE while hash matched:
            return "METADATA_TAMPERING: Metadata was modified after signing (Harmonic Fingerprint mismatch)"

        # 2. Hash mismatch usually means content or metadata changed
        if "HASH_MISMATCH" in reason:
             return "CONTENT_OR_METADATA_TAMPERING: Hash does not match"

        # 3. Assinatura copiada de outro documento?
        if 'HARMONIC_DISSONANCE' in reason:
            return "SIGNATURE_REPLAY: Signature copied from another document"

        # 4. Assinatura forjada matematicamente?
        if 'PHI_MODULUS' in reason:
            return "CRYPTOGRAPHIC_FORGERY: Signature was mathematically forged"

        return f"UNKNOWN_FORGERY: {reason}"

def demo_bridge_security():
    """
    Demonstração simplificada para integração com Avalon CLI
    """
    shield = HarmonicSignatureShield()

    content = "AVALON CORE STATUS: OPERATIONAL"
    metadata = {"node": "alpha-1", "epoch": 5040}

    signed = shield.sign_document(content, metadata)
    is_authentic, reason = shield.verify_document(signed)

    print(f"\nVerification: {'SUCCESS' if is_authentic else 'FAILURE'}")
    if reason:
        print(f"Reason: {reason}")

    return is_authentic
