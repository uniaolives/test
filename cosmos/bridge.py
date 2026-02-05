# cosmos/bridge.py - Ceremony & Biometric Integration
import time
import math
import random
from typing import Dict, List, Any
import numpy as np
from cosmos.core import SingularityNavigator
from cosmos.network import WormholeNetwork, QuantumTimeChain

class CeremonyEngine:
    """Manages the 'Traversal Ceremony' by syncing system state with real-world signals."""
    def __init__(self, duration=144):
        self.duration = duration
        self.start_time = None
        self.active = False

    def start(self):
        """Starts the ceremony cycle."""
        self.start_time = time.time()
        self.active = True
        return "CEREMONY INITIATED: Target Duration = {} seconds".format(self.duration)

    def get_progress(self):
        """Returns the current progress (0.0 to 1.0)."""
        if not self.active or self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return min(elapsed / self.duration, 1.0)

    def complete(self):
        """Completes the ceremony."""
        self.active = False
        return "CEREMONY COMPLETE"

class AdvancedCeremonyEngine(CeremonyEngine):
    """Integrated engine with network and navigator support."""
    def __init__(self, duration=144, node_count=12):
        super().__init__(duration)
        self.navigator = SingularityNavigator()
        self.network = WormholeNetwork(node_count)
        # Mock fidelity matrix as it's used in extract_network_state
        self.network.fidelity_matrix = np.eye(node_count)

    def execute_ceremony_cycle(self):
        """Simulates one cycle of the ceremony."""
        sigma = self.navigator.measure_state()
        status = self.navigator.navigate()
        curvature = self.network.calculate_curvature(0, 8)
        return {
            "sigma": sigma,
            "status": status,
            "curvature": curvature,
            "tau": self.navigator.tau
        }

    def generate_ceremony_report(self, result):
        """Generates a summary report of the cycle."""
        return f"Status: {result['status']} | Sigma: {result['sigma']:.3f} | Curvature: {result['curvature']}"

class TimeLockCeremonyEngine:
    """Motor de cerimônia integrado com qTimeChain."""

    def __init__(self, base_ceremony_engine):
        self.base_engine = base_ceremony_engine
        self.timechain = QuantumTimeChain()

        # Criar bloco gênese com estado inicial
        genesis_data = {
            'network': self._extract_network_state(),
            'ceremony': self._extract_ceremony_state()
        }
        self.timechain.create_genesis_block(genesis_data)

        self.ceremony_start_time = time.time()
        self.last_block_time = time.time()
        self.block_interval = 5.0  # Segundos entre blocos

    def _extract_network_state(self) -> Dict:
        """Extrai estado atual da rede."""
        network = self.base_engine.network
        return {
            'nodes': network.nodes.copy(),
            'edges': network.edges.copy(),
            'fidelity_matrix': network.fidelity_matrix.tolist()
            if hasattr(network.fidelity_matrix, 'tolist')
            else network.fidelity_matrix
        }

    def _extract_ceremony_state(self) -> Dict:
        """Extrai estado atual da cerimônia."""
        nav = self.base_engine.navigator
        return {
            'sigma': nav.sigma,
            'tau': nav.tau,
            'potential': nav.calculate_potential() if hasattr(nav, 'calculate_potential') else 0
        }

    def execute_time_locked_ceremony(self, duration_seconds: float = 60.0):
        """Executa cerimônia com registro em cadeia temporal."""
        print("🔗 INITIATING TIME-LOCKED CEREMONY")
        print(f"   Duration: {duration_seconds} seconds")
        print(f"   Block Interval: {self.block_interval} seconds")
        print("-" * 50)

        start_time = time.time()
        end_time = start_time + duration_seconds
        next_block_time = start_time + self.block_interval

        cycle_count = 0

        try:
            while time.time() < end_time:
                current_time = time.time()
                cycle_count += 1

                # Executar ciclo normal da cerimônia
                ceremony_result = self.base_engine.execute_ceremony_cycle()

                # Verificar se é hora de adicionar novo bloco
                if current_time >= next_block_time:
                    network_state = self._extract_network_state()
                    ceremony_state = self._extract_ceremony_state()

                    new_block = self.timechain.add_block(network_state, ceremony_state)

                    print(f"\n⛓️ New Block #{len(self.timechain.chain)}")
                    print(f"   Hash: {new_block.hash[:16]}...")
                    print(f"   Singularity Score: {new_block.singularity_score:.3f}")

                    next_block_time = current_time + self.block_interval

                # Relatório periódico
                if cycle_count % 20 == 0:
                    print(self.base_engine.generate_ceremony_report(ceremony_result))

                time.sleep(0.01) # Faster than original 0.05 for demo

        except KeyboardInterrupt:
            print("\n⚠️ Ceremony interrupted by operator")

        self._generate_final_report(start_time, cycle_count)

    def _generate_final_report(self, start_time: float, cycle_count: int):
        """Gera relatório final da cerimônia."""
        duration = time.time() - start_time

        print("\n" + "=" * 60)
        print("TIME-LOCKED CEREMONY COMPLETE")
        print("=" * 60)

        print(f"\n📊 CEREMONY METRICS:")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Cycles: {cycle_count}")
        print(f"   Blocks Created: {len(self.timechain.chain)}")

        print(self.timechain.generate_timelock_report())

def visualize_timechain_html(timechain: QuantumTimeChain, filename: str = "timechain_viz.html"):
    """Gera visualização HTML da cadeia temporal."""

    if not timechain.chain:
        return "No chain data to visualize"

    # Preparar dados
    blocks = timechain.chain
    timestamps = [block.timestamp for block in blocks]
    scores = [block.singularity_score for block in blocks]
    hashes = [block.hash for block in blocks]

    # Converter timestamps para horas/minutos/segundos relativos
    start_time = timestamps[0]
    relative_times = [(t - start_time) for t in timestamps]

    def generate_block_list(blocks: List[Any]) -> str:
        """Gera lista HTML de blocos."""
        html_blocks = []
        for i, block in enumerate(blocks):
            block_data = block.to_dict()
            html_blocks.append(f"""
            <div class="block">
                <div>Block #{i+1} | <span class="time">{block_data['human_time']}</span></div>
                <div>Score: <span class="score">{block_data['singularity_score']:.3f}</span></div>
                <div class="hash">Hash: {block_data['hash'][:24]}...</div>
                <div class="hash">Prev: {block_data['previous_hash'][:24]}...</div>
            </div>
            """)
        return "".join(html_blocks)

    # Gerar HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quantum TimeChain Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Courier New', monospace; margin: 20px; background: #0a0a0a; color: #00ff00; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; padding: 20px; border-bottom: 1px solid #00ff00; }}
            .plots {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
            .plot-container {{ flex: 1; min-width: 500px; background: #111; padding: 15px; border-radius: 5px; }}
            .chain-info {{ margin-top: 30px; padding: 20px; background: #111; border-radius: 5px; }}
            .block {{ padding: 10px; margin: 5px 0; background: #222; border-left: 3px solid #00ff00; }}
            .block:hover {{ background: #333; }}
            .hash {{ font-size: 12px; color: #888; }}
            .score {{ color: #ff6b6b; font-weight: bold; }}
            .time {{ color: #64b5f6; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌀 Quantum TimeChain</h1>
                <p>Blocks: {len(blocks)} | Time Span: {relative_times[-1]:.1f}s | Avg Score: {np.mean(scores):.3f}</p>
            </div>

            <div class="plots">
                <div class="plot-container">
                    <div id="scorePlot"></div>
                </div>
                <div class="plot-container">
                    <div id="phasePlot"></div>
                </div>
            </div>

            <div class="chain-info">
                <h3>Latest Blocks (Last 10)</h3>
                {generate_block_list(blocks[-10:]) if len(blocks) >= 10 else generate_block_list(blocks)}
            </div>
        </div>

        <script>
            // Dados para Plotly
            const times = {relative_times};
            const scores = {scores};
            const phases = {[block.time_crystal_phase for block in blocks]};
            const hashes = {hashes};

            // Gráfico 1: Singularity Score vs Time
            Plotly.newPlot('scorePlot', [{{
                x: times,
                y: scores,
                mode: 'lines+markers',
                name: 'Singularity Score',
                line: {{color: '#00ff00', width: 2}},
                marker: {{size: 6, color: scores.map(s => s > 0.9 ? '#ff0000' : '#00ff00')}}
            }}], {{
                title: 'Singularity Score Evolution',
                xaxis: {{title: 'Time (seconds)'}},
                yaxis: {{title: 'Score', range: [0, 1]}},
                plot_bgcolor: '#111',
                paper_bgcolor: '#111',
                font: {{color: '#00ff00'}}
            }});

            // Gráfico 2: Time Crystal Phase
            Plotly.newPlot('phasePlot', [{{
                x: times,
                y: phases,
                mode: 'lines+markers',
                name: 'Time Crystal Phase',
                line: {{color: '#64b5f6', width: 2}},
                marker: {{size: 6}}
            }}], {{
                title: 'Time Crystal Phase Evolution',
                xaxis: {{title: 'Time (seconds)'}},
                yaxis: {{title: 'Phase (radians)'}},
                plot_bgcolor: '#111',
                paper_bgcolor: '#111',
                font: {{color: '#00ff00'}}
            }});
        </script>
    </body>
    </html>
    """

    try:
        with open(filename, 'w') as f:
            f.write(html_content)
        return f"Visualization saved to {filename}"
    except Exception as e:
        return f"Error saving visualization: {str(e)}"

def schumann_generator(n: int = 1) -> float:
    """
    Returns the n-th mode of the Schumann resonance frequency.
    """
    modes = {
        1: 7.83,
        2: 14.1,
        3: 20.3,
        "phi": 16.2
    }
    return modes.get(n, 7.83)

def biometric_simulator() -> dict:
    """
    Simulates biometric input signals for the ceremony.
    """
    return {
        "heart_rate": 60 + random.random() * 20,
        "coherence": 0.5 + random.random() * 0.5,
        "schumann_sync": 0.9 + random.random() * 0.1
    }
