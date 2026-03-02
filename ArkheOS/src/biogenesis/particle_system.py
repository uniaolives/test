"""
PARTICLE SYSTEM v3.0 - Motor de Simulação com Spatial Hashing
Complexidade O(N) para detecção de colisões e interações
"""

import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any

try:
    from .bio_arkhe import BioAgent, ArkheGenome
    from .morphogenetic_field import MorphogeneticField
    from .constraint_engine import ConstraintLearner
except ImportError:
    try:
        from bio_arkhe import BioAgent, ArkheGenome
        from morphogenetic_field import MorphogeneticField
        from constraint_engine import ConstraintLearner
    except ImportError:
        pass


class SpatialHash:
    def __init__(self, cell_size: float = 5.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
    def _get_cell(self, position: np.ndarray) -> Tuple[int, ...]:
        return tuple((position / self.cell_size).astype(int))
    def insert(self, agent_id: int, position: np.ndarray) -> None:
        cell = self._get_cell(position)
        self.grid[cell].add(agent_id)
    def query(self, position: np.ndarray, radius: float) -> List[int]:
        center_cell = self._get_cell(position)
        radius_cells = int(np.ceil(radius / self.cell_size))
        neighbors = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    cell_key = (center_cell[0] + dx, center_cell[1] + dy, center_cell[2] + dz)
                    neighbors.extend(self.grid.get(cell_key, []))
        return neighbors
    def clear(self) -> None:
        self.grid.clear()


class BioGenesisEngine:
    def __init__(self, num_agents: int = 150):
        self.field = MorphogeneticField((100, 100, 100))
        self.spatial_hash = SpatialHash(cell_size=5.0)
        self.agents: Dict[int, BioAgent] = {}
        self.next_id = 0
        self.simulation_time = 0.0
        self.signal_sources = []
        self.stats = {'births': 0, 'deaths': 0, 'bonds_formed': 0, 'bonds_broken': 0}
        self._initialize_population(num_agents)
        self._add_initial_signal_sources()

    def _initialize_population(self, num_agents: int) -> None:
        tribe_centers = [np.array([25.0, 25.0, 50.0]), np.array([50.0, 75.0, 50.0]), np.array([75.0, 50.0, 25.0])]
        base_chemistry = [0.25, 0.50, 0.75]
        for i in range(num_agents):
            tribe = i % 3
            pos = (tribe_centers[tribe] + np.random.randn(3).astype(np.float32) * 15)
            pos = np.clip(pos, 5, 94)
            genome = ArkheGenome(C=float(np.clip(base_chemistry[tribe] + np.random.normal(0, 0.08), 0.1, 0.9)),
                                 I=float(np.random.uniform(0.2, 0.8)), E=float(np.random.uniform(0.4, 1.0)),
                                 F=float(np.random.uniform(0.2, 0.8)))
            agent = BioAgent(self.next_id, pos[0], pos[1], pos[2], genome)
            brain = ConstraintLearner(self.next_id, genome.to_vector())
            agent.set_brain(brain)
            self.agents[self.next_id] = agent
            self.next_id += 1
            self.stats['births'] += 1

    def _add_initial_signal_sources(self) -> None:
        self.add_signal_source(np.array([50.0, 50.0, 50.0]), 25.0, float('inf'))
        for _ in range(3):
            pos = np.random.rand(3) * 60 + 20
            self.add_signal_source(pos, 12.0, 500.0)

    def add_signal_source(self, position: np.ndarray, strength: float, duration: float = 200.0) -> None:
        self.signal_sources.append({'position': position.copy(), 'strength': strength, 'duration': duration})

    def update(self, dt: float = 0.1) -> None:
        self.simulation_time += dt
        self.field.step(dt) # Use step() instead of diffuse_and_decay
        new_sources = []
        for source in self.signal_sources:
            self.field.add_signal(source['position'][0], source['position'][1], source['position'][2], source['strength'])
            if source['duration'] != float('inf'):
                source['duration'] -= 1
                if source['duration'] > 0: new_sources.append(source)
        self.signal_sources = new_sources
        self.spatial_hash.clear()
        for agent in self.agents.values():
            if agent.is_alive(): self.spatial_hash.insert(agent.id, agent.position)
        previous_health = {aid: agent.health for aid, agent in self.agents.items() if agent.is_alive()}
        for agent in list(self.agents.values()):
            if not agent.is_alive(): continue
            signal_strength = self.field.get_signal_at(agent.position[0], agent.position[1], agent.position[2])
            if signal_strength > 2.0:
                gradient = self.field.get_gradient(agent.position[0], agent.position[1], agent.position[2])
                agent.velocity += gradient * agent.genome.E * 0.15
                agent.state = "foraging"
                agent.health = min(1.0, agent.health + 0.002 * agent.genome.E)
            else:
                noise = np.random.randn(3).astype(np.float32) * 0.08
                agent.velocity += noise * agent.genome.E
                agent.state = "exploring"
            if agent.health > 0.6 and agent.genome.F > 0.4:
                self.field.add_signal(agent.position[0], agent.position[1], agent.position[2], agent.genome.F * 0.3)
            agent.apply_physics(dt, self.field.shape)
        self._process_interactions()
        self._apply_learning_feedback(previous_health)
        self._cleanup_dead_agents()

    def _process_interactions(self) -> None:
        processed_pairs = set()
        for agent in list(self.agents.values()):
            if not agent.is_alive(): continue
            nearby_ids = self.spatial_hash.query(agent.position, radius=4.0)
            for other_id in nearby_ids:
                if other_id <= agent.id: continue
                if other_id not in self.agents: continue
                other = self.agents[other_id]
                if not other.is_alive(): continue
                pair = tuple(sorted((agent.id, other_id)))
                if pair in processed_pairs: continue
                processed_pairs.add(pair)
                dist = np.linalg.norm(agent.position - other.position)
                if dist > 3.5: continue
                compatibility = 1.0 - abs(agent.genome.C - other.genome.C)
                if other_id in agent.connections:
                    energy_exchange = (compatibility - 0.5) * 0.012
                    if energy_exchange > 0:
                        agent.health = min(1.0, agent.health + energy_exchange)
                        other.health = min(1.0, other.health + energy_exchange)
                        agent.bond_strengths[other_id] = min(agent.bond_strengths.get(other_id, 0.5) + 0.008, 1.0)
                        agent.state = "socializing"
                    else:
                        agent.health += energy_exchange
                        other.health += energy_exchange
                else:
                    if len(agent.connections) < 6 and len(other.connections) < 6:
                        if agent.brain and other.brain:
                            score_a, _ = agent.brain.evaluate_partner(other.genome, self.simulation_time)
                            score_b, _ = other.brain.evaluate_partner(agent.genome, self.simulation_time)
                            if score_a > 0.05 and score_b > 0.05:
                                success = agent.form_bond(other, strength=(score_a + score_b) / 2)
                                if success:
                                    self.stats['bonds_formed'] += 1
                                    agent.brain.learn_from_experience(other.genome, 0.1, self.simulation_time)
                                    other.brain.learn_from_experience(agent.genome, 0.1, self.simulation_time)
                            else:
                                if score_a < -0.2: agent.brain.learn_from_experience(other.genome, -0.05, self.simulation_time)
                                if score_b < -0.2: other.brain.learn_from_experience(agent.genome, -0.05, self.simulation_time)

    def _apply_learning_feedback(self, previous_health: Dict[int, float]) -> None:
        for agent_id, agent in self.agents.items():
            if not agent.is_alive() or not agent.brain: continue
            delta_health = agent.health - previous_health.get(agent_id, agent.health)
            if abs(delta_health) > 0.0005 and agent.connections:
                share = delta_health / len(agent.connections)
                for neighbor_id in agent.connections:
                    if neighbor_id in self.agents:
                        neighbor = self.agents[neighbor_id]
                        if neighbor.is_alive() and neighbor.brain:
                            agent.brain.learn_from_experience(neighbor.genome, share, self.simulation_time)

    def _cleanup_dead_agents(self) -> None:
        dead_ids = [aid for aid, agent in self.agents.items() if not agent.is_alive()]
        for aid in dead_ids:
            agent = self.agents[aid]
            for neighbor_id in list(agent.connections):
                if neighbor_id in self.agents:
                    neighbor = self.agents[neighbor_id]
                    neighbor.break_bond(aid)
                    if neighbor.brain: neighbor.brain.learn_from_experience(agent.genome, -0.2, self.simulation_time)
            del self.agents[aid]
            self.stats['deaths'] += 1

    def get_stats(self) -> Dict[str, Any]:
        alive = [a for a in self.agents.values() if a.is_alive()]
        return {'agents': len(alive), 'time': round(float(self.simulation_time), 1), 'bonds': int(self.stats['bonds_formed']),
                'deaths': int(self.stats['deaths']), 'avg_health': round(float(np.mean([a.health for a in alive])), 3) if alive else 0.0}

    def get_agent_info(self, agent_id: int) -> Optional[Dict[str, Any]]:
        if agent_id not in self.agents: return None
        agent = self.agents[agent_id]
        if not agent.is_alive(): return None
        info = {'id': int(agent.id), 'position': agent.get_position(), 'health': round(float(agent.health), 3),
                'age': round(float(agent.age), 1), 'state': str(agent.state),
                'genome': {'C': round(float(agent.genome.C), 2), 'I': round(float(agent.genome.I), 2),
                           'E': round(float(agent.genome.E), 2), 'F': round(float(agent.genome.F), 2)},
                'connections': int(len(agent.connections)),
                'profile': (agent.brain.get_cognitive_profile() if agent.brain else "N/A")}
        return info
