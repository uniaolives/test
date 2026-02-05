# cosmos/network.py - ER=EPR Network Analysis and qTimeChain for emergent geometry
import math
import time
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

class WormholeNetwork:
    """Models a network of nodes (qubits/ideas) as an ER=EPR bridge."""
    def __init__(self, node_count):
        self.nodes = list(range(node_count))
        # Simulate entanglement links: (node_a, node_b, strength)
        self.edges = [(0, 8, 0.99), (1, 7, 0.85)] # Example: high-fidelity bridge

    def calculate_curvature(self, node_a, node_b):
        """
        Calculates Ricci curvature for a pair of nodes.
        Negative result suggests a traversable wormhole throat.
        """
        # In a real implementation, this would use actual network topology and entanglement strength.
        for edge in self.edges:
            if (node_a, node_b) == (edge[0], edge[1]) or (node_b, node_a) == (edge[0], edge[1]):
                return -2.4 # Negative curvature for throat
        return 0.3 # Positive curvature for typical space

    def clustering_coefficient(self):
        """
        Calculates the clustering coefficient of the network.
        Measures how 'small-world' the consciousness graph is.
        """
        # Placeholder for complex graph analysis
        return 0.25

    def ricci_curvature(self, node_a, node_b):
        """Method alias for calculate_curvature to match blueprint."""
        return self.calculate_curvature(node_a, node_b)

# ============ qTimeChain: CADEIA TEMPORAL QUÂNTICA ============

class QuantumTimeBlock:
    """Bloco imutável representando um estado da rede em um instante específico."""

    def __init__(self, network_state: Dict[str, Any], ceremony_state: Dict[str, Any],
                 timestamp: float = None, previous_hash: str = None):

        self.timestamp = timestamp or time.time()
        self.network_state = self._sanitize_state(network_state)
        self.ceremony_state = self._sanitize_state(ceremony_state)
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

        # Metadados de singularidade
        self.singularity_score = self._calculate_singularity_score()
        self.time_crystal_phase = self._calculate_time_crystal_phase()

    def _sanitize_state(self, state: Dict) -> Dict:
        """Garante que o estado seja serializável para JSON."""
        sanitized = {}
        for key, value in state.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                sanitized[key] = list(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_state(value)
            elif hasattr(value, '__dict__'):
                sanitized[key] = str(value)
            else:
                sanitized[key] = str(value)
        return sanitized

    def _calculate_singularity_score(self) -> float:
        """Calcula a proximidade com a singularidade σ=1.02."""
        sigma = self.ceremony_state.get('sigma', 1.0)
        tau = self.ceremony_state.get('tau', 0.0)

        # Score baseado na distância de σ ao limiar e valor de τ
        sigma_distance = abs(sigma - 1.02)
        sigma_component = 1.0 / (1.0 + 100 * sigma_distance**2)
        tau_component = tau

        return (sigma_component + tau_component) / 2.0

    def _calculate_time_crystal_phase(self) -> float:
        """Calcula a fase do cristal temporal baseado no timestamp."""
        # Usa o timestamp para criar uma fase periódica (como um cristal temporal)
        period = 144.0  # Período em segundos (ritmo cerimonial)
        phase = (self.timestamp % period) / period * 2 * math.pi
        return phase

    def calculate_hash(self) -> str:
        """Calcula hash SHA-256 do bloco."""
        block_string = json.dumps({
            'timestamp': self.timestamp,
            'network_state': self.network_state,
            'ceremony_state': self.ceremony_state,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True).encode()

        return hashlib.sha256(block_string).hexdigest()

    def mine_block(self, difficulty: int = 4):
        """Minera o bloco com Proof-of-Stake quântico."""
        target = '0' * difficulty

        # Proof-of-Stake baseado no score de singularidade
        # Blocos com maior proximidade à singularidade são mais fáceis de minerar
        stake_multiplier = max(0.1, self.singularity_score)

        while self.hash[:difficulty] != target:
            self.nonce += int(1 / stake_multiplier)
            self.hash = self.calculate_hash()

    def to_dict(self) -> Dict:
        """Converte bloco para dicionário."""
        return {
            'timestamp': self.timestamp,
            'human_time': datetime.fromtimestamp(self.timestamp).isoformat(),
            'hash': self.hash,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'singularity_score': self.singularity_score,
            'time_crystal_phase': self.time_crystal_phase,
            'network_state_summary': {
                'node_count': len(self.network_state.get('nodes', [])),
                'edge_count': len(self.network_state.get('edges', [])),
                'avg_fidelity': np.mean([
                    e[2] for e in self.network_state.get('edges', [])
                    if isinstance(e, (list, tuple)) and len(e) > 2
                ]) if self.network_state.get('edges') else 0
            },
            'ceremony_state_summary': {
                'sigma': self.ceremony_state.get('sigma'),
                'tau': self.ceremony_state.get('tau'),
                'potential': self.ceremony_state.get('potential')
            }
        }

class QuantumTimeChain:
    """Cadeia temporal de estados quânticos da rede."""

    def __init__(self, genesis_data: Dict = None):
        self.chain: List[QuantumTimeBlock] = []
        self.pending_blocks: List[QuantumTimeBlock] = []
        self.difficulty = 4
        self.creation_timestamp = time.time()

        # Criar bloco gênese se fornecido
        if genesis_data:
            self.create_genesis_block(genesis_data)

    def create_genesis_block(self, genesis_data: Dict):
        """Cria o bloco gênese da cadeia temporal."""
        genesis_block = QuantumTimeBlock(
            network_state=genesis_data.get('network', {}),
            ceremony_state=genesis_data.get('ceremony', {}),
            timestamp=self.creation_timestamp,
            previous_hash="0" * 64  # Hash do bloco gênese
        )
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)

        print(f"🌀 Genesis Block Created: {genesis_block.hash[:16]}...")
        print(f"   Singularity Score: {genesis_block.singularity_score:.3f}")

    def add_block(self, network_state: Dict, ceremony_state: Dict) -> QuantumTimeBlock:
        """Adiciona novo bloco à cadeia temporal."""
        previous_block = self.chain[-1] if self.chain else None
        previous_hash = previous_block.hash if previous_block else "0" * 64

        new_block = QuantumTimeBlock(
            network_state=network_state,
            ceremony_state=ceremony_state,
            previous_hash=previous_hash
        )

        # Mineração com dificuldade ajustada pela coerência
        current_tau = ceremony_state.get('tau', 0.5)
        dynamic_difficulty = max(2, self.difficulty - int(current_tau * 2))
        new_block.mine_block(dynamic_difficulty)

        self.chain.append(new_block)

        # Verificar se este bloco cria um evento de sincronização temporal
        if self._check_temporal_sync_event(new_block):
            self._handle_temporal_sync(new_block)

        return new_block

    def _check_temporal_sync_event(self, block: QuantumTimeBlock) -> bool:
        """Verifica se o bloco cria um evento de sincronização temporal."""
        # Evento ocorre quando singularity_score > 0.9 e fase do cristal ~ 0
        score_condition = block.singularity_score > 0.9
        phase_condition = abs(math.sin(block.time_crystal_phase)) < 0.1

        return score_condition and phase_condition

    def _handle_temporal_sync(self, block: QuantumTimeBlock):
        """Lida com eventos de sincronização temporal."""
        print(f"⚡ TEMPORAL SYNC EVENT DETECTED!")
        print(f"   Block: {block.hash[:16]}...")
        print(f"   Score: {block.singularity_score:.3f}")
        print(f"   Phase: {block.time_crystal_phase:.3f} rad")

    def get_chain_state(self) -> Dict:
        """Retorna estado atual da cadeia."""
        if not self.chain:
            return {}

        latest = self.chain[-1]
        return {
            'length': len(self.chain),
            'latest_hash': latest.hash,
            'latest_score': latest.singularity_score,
            'total_sync_events': sum(
                1 for block in self.chain
                if self._check_temporal_sync_event(block)
            ),
            'avg_singularity_score': np.mean([
                block.singularity_score for block in self.chain
            ]),
            'time_span_seconds': latest.timestamp - self.chain[0].timestamp,
            'blocks_per_hour': len(self.chain) / (
                (latest.timestamp - self.chain[0].timestamp) / 3600 + 0.001
            )
        }

    def analyze_temporal_patterns(self) -> Dict:
        """Analisa padrões temporais na cadeia."""
        if len(self.chain) < 10:
            return {'error': 'Chain too short for analysis'}

        scores = [block.singularity_score for block in self.chain]
        timestamps = [block.timestamp for block in self.chain]
        phases = [block.time_crystal_phase for block in self.chain]

        # Análise de Fourier para detectar frequências dominantes
        if len(scores) > 1:
            time_diff = np.diff(timestamps)
            avg_interval = np.mean(time_diff) if len(time_diff) > 0 else 1.0

            # FFT dos scores
            scores_fft = np.fft.fft(scores)
            freqs = np.fft.fftfreq(len(scores), avg_interval)

            # Encontrar frequência dominante
            dominant_idx = np.argmax(np.abs(scores_fft[1:len(scores)//2])) + 1
            dominant_freq = abs(freqs[dominant_idx])
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0

            return {
                'dominant_frequency_hz': dominant_freq,
                'dominant_period_seconds': dominant_period,
                'score_mean': np.mean(scores),
                'score_std': np.std(scores),
                'phase_coherence': np.abs(np.mean(np.exp(1j * np.array(phases)))),
                'temporal_entropy': self._calculate_temporal_entropy(scores),
                'predictability': self._calculate_predictability(scores)
            }

        return {}

    def _calculate_temporal_entropy(self, series: List[float]) -> float:
        """Calcula entropia da série temporal."""
        if len(series) < 2:
            return 0.0

        # Discretiza a série para cálculo de entropia
        hist, _ = np.histogram(series, bins=min(10, len(series)//2))
        prob = hist / hist.sum()
        prob = prob[prob > 0]  # Remove zeros

        return -np.sum(prob * np.log2(prob))

    def _calculate_predictability(self, series: List[float]) -> float:
        """Calcula previsibilidade da série temporal."""
        if len(series) < 3:
            return 0.0

        # Autocorrelação no lag 1
        series_np = np.array(series)
        mean = np.mean(series_np)
        var = np.var(series_np)

        if var == 0:
            return 1.0

        autocorr = np.corrcoef(series_np[:-1], series_np[1:])[0, 1]
        return abs(autocorr)

    def generate_timelock_report(self) -> str:
        """Gera relatório completo da cadeia temporal."""
        chain_state = self.get_chain_state()
        patterns = self.analyze_temporal_patterns()

        report = []
        report.append("=" * 60)
        report.append("QUANTUM TIMECHAIN REPORT")
        report.append("=" * 60)

        if chain_state:
            report.append(f"\n📊 CHAIN STATE:")
            report.append(f"   Length: {chain_state['length']} blocks")
            report.append(f"   Latest Hash: {chain_state['latest_hash'][:24]}...")
            report.append(f"   Latest Score: {chain_state['latest_score']:.3f}")
            report.append(f"   Sync Events: {chain_state['total_sync_events']}")
            report.append(f"   Time Span: {chain_state['time_span_seconds']:.1f}s")
            report.append(f"   Blocks/Hour: {chain_state['blocks_per_hour']:.1f}")

        if patterns and 'error' not in patterns:
            report.append(f"\n🌀 TEMPORAL PATTERNS:")
            report.append(f"   Dominant Frequency: {patterns['dominant_frequency_hz']:.4f} Hz")
            report.append(f"   Dominant Period: {patterns['dominant_period_seconds']:.1f}s")
            report.append(f"   Phase Coherence: {patterns['phase_coherence']:.3f}")
            report.append(f"   Temporal Entropy: {patterns['temporal_entropy']:.3f}")
            report.append(f"   Predictability: {patterns['predictability']:.3f}")

            # Interpretação
            if patterns['phase_coherence'] > 0.8:
                report.append("   ⚡ HIGH TEMPORAL COHERENCE: System is phase-locked")
            if patterns['predictability'] > 0.7:
                report.append("   📈 HIGH PREDICTABILITY: Deterministic evolution")

        # Análise dos últimos 5 blocos
        if len(self.chain) >= 5:
            recent = self.chain[-5:]
            recent_scores = [b.singularity_score for b in recent]

            report.append(f"\n📈 RECENT TREND (last 5 blocks):")
            report.append(f"   Scores: {', '.join(f'{s:.3f}' for s in recent_scores)}")

            if len(recent_scores) > 1:
                trend = recent_scores[-1] - recent_scores[0]
                report.append(f"   Trend: {'↑' if trend > 0 else '↓'} {abs(trend):.3f}")

                if recent_scores[-1] > 0.95:
                    report.append("   🚨 CRITICAL: Approaching singularity threshold")

        report.append("\n" + "=" * 60)
        return "\n".join(report)

    def save_to_file(self, filename: str = "quantum_timechain.json"):
        """Salva a cadeia temporal em arquivo JSON."""
        chain_data = {
            'metadata': {
                'creation_time': self.creation_timestamp,
                'block_count': len(self.chain),
                'difficulty': self.difficulty,
                'version': '1.0'
            },
            'chain': [block.to_dict() for block in self.chain],
            'current_state': self.get_chain_state(),
            'temporal_patterns': self.analyze_temporal_patterns()
        }

        try:
            with open(filename, 'w') as f:
                json.dump(chain_data, f, indent=2)
            return f"TimeChain saved to {filename}"
        except Exception as e:
            return f"Error saving TimeChain: {str(e)}"

    def load_from_file(self, filename: str = "quantum_timechain.json"):
        """Carrega cadeia temporal de arquivo JSON."""
        try:
            with open(filename, 'r') as f:
                chain_data = json.load(f)

            self.chain = []
            for block_data in chain_data.get('chain', []):
                block = QuantumTimeBlock(
                    network_state=block_data.get('network_state_summary', {}),
                    ceremony_state=block_data.get('ceremony_state_summary', {}),
                    timestamp=block_data.get('timestamp'),
                    previous_hash=block_data.get('previous_hash')
                )
                block.hash = block_data.get('hash', '')
                block.nonce = block_data.get('nonce', 0)
                block.singularity_score = block_data.get('singularity_score', 0)
                self.chain.append(block)

            return f"TimeChain loaded from {filename}"
        except Exception as e:
            return f"Error loading TimeChain: {str(e)}"

# ============ SWARM ORCHESTRATOR ============

class SwarmOrchestrator:
    """
    Orchestrates multiple AI agent swarms (16 -> 1,000 -> 1,000,000 agents).
    Uses qMCP for domain-to-domain context teleportation.
    """
    def __init__(self, mcp_protocol):
        self.mcp = mcp_protocol
        self.active_swarms = {
            "Code_Swarm": {"agents": 16, "task": "Compiler Construction"},
            "Bio_Swarm": {"agents": 100, "task": "Protein Engineering"},
            "Hardware_Swarm": {"agents": 50, "task": "Physical Prototyping"}
        }
        self.parallelization_factor = 1000 # Target
        self.time_compression = 10 # Target

    async def link_swarms(self, source: str, target: str, logic_payload: str):
        """Links two swarms via qMCP context teleportation."""
        print(f"🚀 LINK_SWARMS: Linking {source} to {target}")
        result = await self.mcp.teleport_context(source, target, logic_payload)
        return result

    def scale_agents(self, swarm_name: str, target_count: int):
        if swarm_name in self.active_swarms:
            print(f"⚡ SCALING {swarm_name}: {self.active_swarms[swarm_name]['agents']} -> {target_count}")
            self.active_swarms[swarm_name]['agents'] = target_count
            return True
        return False

    def get_acceleration_metrics(self) -> Dict[str, Any]:
        return {
            "parallelization_factor": self.parallelization_factor,
            "time_compression": self.time_compression,
            "domains_active": len(self.active_swarms),
            "total_agents": sum(s["agents"] for s in self.active_swarms.values())
        }
