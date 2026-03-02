# arkhe/crypto/pq_arkhe.py
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import hashlib
import secrets

# Importações simuladas do Kyber (em produção: pycryptodome ou liboqs)
class Kyber512:
    @staticmethod
    def keygen():
        # Simulação: retorna (pk, sk) aleatórios
        pk = secrets.token_bytes(800)
        sk = secrets.token_bytes(1632)
        return pk, sk

    @staticmethod
    def encapsulate(pk: bytes):
        # Simulação: retorna (ct, shared_secret)
        ct = secrets.token_bytes(768)
        ss = b"fixed_secret_for_simulation_32b"
        return ct, ss

    @staticmethod
    def decapsulate(ct: bytes, sk: bytes):
        return b"fixed_secret_for_simulation_32b"

@dataclass
class CorrelationProfile:
    """
    Perfil de correlação entre dois nós no manifold.
    Representa o "estado emaranhado" clássico.
    """
    node_id_a: str
    node_id_b: str

    # Histogramas de latência (distribuição de probabilidade)
    latency_hist: np.ndarray  # bins de tempo de resposta

    # Padrões de jitter (variância temporal)
    jitter_signature: np.ndarray

    # Entropia de transferência (direcionalidade)
    transfer_entropy: float

    # Coeficiente de correlação de Pearson histórico
    correlation_coeff: float

    # Timestamp de estabelecimento
    established_at: float

    # "Fidelidade" do canal (análogo à fidelidade quântica)
    fidelity: float

class QuantumSafeChannel:
    """
    Canal criptográfico híbrido: Kyber para confidencialidade,
    Arkhe Correlation Monitor para detecção de MITM.
    """

    # Thresholds de anomalia (ajustáveis)
    ANOMALY_LATENCY_SIGMA = 3.0
    ANOMALY_CORRELATION_DROP = 0.3
    ANOMALY_ENTROPY_SPIKE = 2.0

    def __init__(self, node_id: str, peer_id: str):
        self.node_id = node_id
        self.peer_id = peer_id

        # Estado Kyber
        self.kyber_pk: Optional[bytes] = None
        self.kyber_sk: Optional[bytes] = None
        self.shared_secret: Optional[bytes] = None

        # Perfil de correlação (inicializado durante handshake)
        self.profile: Optional[CorrelationProfile] = None

        # Janela de observação para detecção em tempo real
        self.observation_window: List[dict] = []
        self.window_size = 100

        # Estado do canal
        self.is_established = False
        self.anomaly_score = 0.0

    def initiate_handshake(self) -> dict:
        """
        Fase 1: Gerar chaves Kyber e iniciar troca.
        """
        self.kyber_pk, self.kyber_sk = Kyber512.keygen()

        # Enviar PK + nonce para medição de latência
        nonce = secrets.token_bytes(32)

        return {
            'type': 'ARKHE_HANDSHAKE_INIT',
            'from': self.node_id,
            'kyber_pk': self.kyber_pk.hex(),
            'nonce': nonce.hex(),
            'timestamp': np.datetime64('now').astype(float)
        }

    def respond_handshake(self, init_msg: dict) -> dict:
        """
        Fase 2: Responder com encapsulamento Kyber + perfil inicial.
        """
        peer_pk = bytes.fromhex(init_msg['kyber_pk'])

        # Encapsulamento Kyber
        ct, ss = Kyber512.encapsulate(peer_pk)
        self.shared_secret = ss

        # Medir latência de ida (para perfil)
        t_sent = init_msg['timestamp']
        t_received = np.datetime64('now').astype(float)
        latency_outbound = t_received - t_sent

        # Gerar perfil inicial (será refinado)
        self.profile = CorrelationProfile(
            node_id_a=self.node_id,
            node_id_b=init_msg['from'],
            latency_hist=self._init_histogram(latency_outbound),
            jitter_signature=np.zeros(10),
            transfer_entropy=0.5,  # neutro inicialmente
            correlation_coeff=0.0,  # será calculado
            established_at=t_received,
            fidelity=1.0
        )

        return {
            'type': 'ARKHE_HANDSHAKE_RESP',
            'from': self.node_id,
            'kyber_ct': ct.hex(),
            'latency_measured': latency_outbound,
            'timestamp': t_received
        }

    def complete_handshake(self, resp_msg: dict) -> bool:
        """
        Fase 3: Completar handshake, decapsular, estabelecer perfil.
        """
        ct = bytes.fromhex(resp_msg['kyber_ct'])
        ss = Kyber512.decapsulate(ct, self.kyber_sk)
        self.shared_secret = ss

        # Calcular latência de volta
        t_roundtrip = np.datetime64('now').astype(float) - resp_msg['timestamp']

        # Estabelecer perfil de correlação bidirecional
        self.profile = CorrelationProfile(
            node_id_a=self.node_id,
            node_id_b=resp_msg['from'],
            latency_hist=self._init_histogram(t_roundtrip / 2),  # estimativa one-way
            jitter_signature=self._calculate_jitter([]),  # será populado
            transfer_entropy=0.5,
            correlation_coeff=1.0,  # perfeito consigo mesmo inicialmente
            established_at=np.datetime64('now').astype(float),
            fidelity=1.0
        )

        self.is_established = True
        return True

    def _init_histogram(self, initial_value: float, bins: int = 50) -> np.ndarray:
        """Inicializa histograma de latência centrado no valor inicial."""
        hist = np.zeros(bins)
        center = int(bins / 2)
        hist[center] = 1.0
        return hist

    def _calculate_jitter(self, samples: List[float]) -> np.ndarray:
        """Calcula assinatura de jitter via FFT das diferenças."""
        if len(samples) < 2:
            return np.zeros(10)
        diffs = np.diff(samples)
        fft = np.fft.fft(diffs, n=10)
        return np.abs(fft)

    def send_message(self, plaintext: bytes) -> dict:
        """
        Envia mensagem com criptografia Kyber + metadados de correlação.
        """
        if not self.is_established:
            raise ValueError("Canal não estabelecido")

        # Criptografar com shared_secret (simulado: XOR simplificado)
        # Em produção: AES-256-GCM com chave derivada de shared_secret
        ciphertext = self._encrypt(plaintext, self.shared_secret)

        # Coletar métricas para perfil
        t_send = np.datetime64('now').astype(float)

        return {
            'type': 'ARKHE_DATA',
            'ciphertext': ciphertext.hex(),
            'timestamp': t_send,
            'seq_num': len(self.observation_window)
        }

    def receive_message(self, msg: dict) -> Tuple[bytes, dict]:
        """
        Recebe mensagem, decriptografa, e verifica anomalias de correlação.
        """
        t_receive = np.datetime64('now').astype(float)

        # Calcular latência observada
        latency = t_receive - msg['timestamp']

        # Atualizar janela de observação
        self.observation_window.append({
            'latency': latency,
            'seq_num': msg['seq_num'],
            'timestamp': t_receive
        })

        if len(self.observation_window) > self.window_size:
            self.observation_window.pop(0)

        # Verificar anomalias
        anomaly_report = self._check_anomalies(latency, msg)

        # Decriptografar
        ciphertext = bytes.fromhex(msg['ciphertext'])
        plaintext = self._decrypt(ciphertext, self.shared_secret)

        return plaintext, anomaly_report

    def _check_anomalies(self, current_latency: float, msg: dict) -> dict:
        """
        Verifica violações de correlação que indicam MITM ou tampering.
        """
        if len(self.observation_window) < 10:
            return {'status': 'INSUFFICIENT_DATA'}

        latencies = [o['latency'] for o in self.observation_window]

        anomalies = []

        # 1. Anomalia de Latência (desvio de média)
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        z_score = abs(current_latency - mean_lat) / (std_lat + 1e-10)

        if z_score > self.ANOMALY_LATENCY_SIGMA:
            anomalies.append({
                'type': 'LATENCY_SPIKE',
                'severity': 'HIGH' if z_score > 5 else 'MEDIUM',
                'z_score': z_score,
                'expected': mean_lat,
                'observed': current_latency
            })

        # 2. Anomalia de Jitter (mudança na assinatura de frequência)
        current_jitter = self._calculate_jitter(latencies)
        if hasattr(self, '_last_jitter'):
            jitter_corr = np.corrcoef(current_jitter, self._last_jitter)[0,1]
            if jitter_corr < self.ANOMALY_CORRELATION_DROP:
                anomalies.append({
                    'type': 'JITTER_SIGNATURE_CHANGE',
                    'severity': 'CRITICAL',
                    'correlation': jitter_corr,
                    'indication': 'Possível interceptação ou rerouting'
                })
        self._last_jitter = current_jitter

        # 3. Anomalia de Entropia (padrão informacional alterado)
        # Calcular entropia de Shannon das latências recentes
        hist, _ = np.histogram(latencies, bins=20, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))

        if hasattr(self, '_baseline_entropy'):
            entropy_spike = abs(entropy - self._baseline_entropy) / self._baseline_entropy
            if entropy_spike > self.ANOMALY_ENTROPY_SPIKE:
                anomalies.append({
                    'type': 'ENTROPY_ANOMALY',
                    'severity': 'HIGH',
                    'entropy_current': entropy,
                    'entropy_baseline': self._baseline_entropy,
                    'indication': 'Comportamento de rede não-natural'
                })
        else:
            self._baseline_entropy = entropy

        # 4. Verificação de sequência (detecção de replay/delay)
        seq_nums = [o['seq_num'] for o in self.observation_window]
        if msg['seq_num'] in seq_nums[:-1]:
            anomalies.append({
                'type': 'REPLAY_ATTACK',
                'severity': 'CRITICAL',
                'seq_num': msg['seq_num']
            })

        # Atualizar score de anomalia agregado
        self.anomaly_score = min(1.0, self.anomaly_score * 0.9 + len(anomalies) * 0.1)

        return {
            'status': 'ANOMALY_DETECTED' if anomalies else 'NORMAL',
            'anomalies': anomalies,
            'anomaly_score': self.anomaly_score,
            'recommendation': 'ROTATE_KEYS' if self.anomaly_score > 0.7 else 'CONTINUE'
        }

    def _encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Simulação: XOR com chave expandida."""
        expanded = (key * (len(plaintext) // len(key) + 1))[:len(plaintext)]
        return bytes(p ^ k for p, k in zip(plaintext, expanded))

    def _decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        return self._encrypt(ciphertext, key)  # XOR é simétrico

    def emergency_rotation(self) -> dict:
        """
        Rotação de chaves forçada quando anomalias são detectadas.
        """
        self.is_established = False
        self.anomaly_score = 0.0
        self.observation_window = []

        # Gerar novas chaves Kyber
        return self.initiate_handshake()

class ArkheSecureNetwork:
    """
    Rede de canais seguros pós-quânticos com monitoramento de correlação.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.channels: dict[str, QuantumSafeChannel] = {}
        self.global_anomaly_map: dict[str, float] = {}

    def establish_channel(self, peer_id: str) -> bool:
        """Estabelece canal seguro com detecção de anomalias."""
        channel = QuantumSafeChannel(self.node_id, peer_id)

        # Simulação de handshake (em produção: troca real de mensagens)
        init = channel.initiate_handshake()
        resp = channel.respond_handshake(init)
        success = channel.complete_handshake(resp)

        if success:
            self.channels[peer_id] = channel
        return success

    def broadcast_secure(self, message: bytes, exclude_anomalous: bool = True):
        """
        Broadcast com roteamento adaptativo baseado em anomalias.
        """
        recipients = []
        for peer_id, channel in self.channels.items():
            if exclude_anomalous and channel.anomaly_score > 0.5:
                continue

            msg = channel.send_message(message)
            recipients.append(peer_id)

            # Atualizar mapa global
            self.global_anomaly_map[peer_id] = channel.anomaly_score

        return {
            'recipients': recipients,
            'anomaly_map': self.global_anomaly_map.copy()
        }

    def detect_global_intrusion(self) -> dict:
        """
        Detecção de intrusão em nível de rede via análise de correlação.
        """
        if len(self.channels) < 3:
            return {'status': 'INSUFFICIENT_NODES'}

        # Matriz de correlação entre todos os pares de canais
        scores = [c.anomaly_score for c in self.channels.values()]
        if len(set(scores)) <= 1: # Avoid issues with zero variance
            corr_matrix = np.eye(len(scores))
        else:
            corr_matrix = np.corrcoef(scores)

        # Se múltiplos canais apresentam anomalias correlacionadas,
        # indica ataque coordenado ou comprometimento de infraestrutura
        correlated_anomalies = np.sum(np.abs(corr_matrix) > 0.8) - len(scores)

        return {
            'status': 'GLOBAL_ANOMALY' if correlated_anomalies > len(scores)/2 else 'NORMAL',
            'correlated_pairs': int(correlated_anomalies),
            'attack_probability': correlated_anomalies / (len(scores)**2 - len(scores) + 1e-10),
            'recommendation': 'ISOLATE_NETWORK' if correlated_anomalies > len(scores)/2 else 'MONITOR'
        }
