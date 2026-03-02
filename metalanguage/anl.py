#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARKHE(N) LANGUAGE (ANL) – Core Python Module (v0.7)
This module provides a Python implementation of the core concepts of the
Arkhe(n) Language: nodes, handovers, hypergraph, protocols, and specialized
constructs for AGI transition (Active Inference, Ouroboros loops).

Author: Based on the Arkhe(n) Language specification.
Date: 2026-02-21
License: CC BY-NC-ND 4.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import time
import hashlib
import json
import threading
import queue
from collections import defaultdict

# ============================================================================
# 1. PRIMITIVAS FUNDAMENTAIS
# ============================================================================

class Protocol(Enum):
    """How a handover preserves or transforms information."""
    CONSERVATIVE = 1   # preserves quantities (energy, coherence)
    CREATIVE = 2       # creates new information or structure (stochastic sampling)
    DESTRUCTIVE = 3    # dissipates or removes (forgetting)
    TRANSMUTATIVE = 4  # changes type (e.g., string → embedding)


@dataclass
class StateSpace:
    """Description of the state space of a node."""
    dimension: int
    topology: str           # "euclidean", "spherical", "hyperbolic", "discrete", etc.
    algebra: str            # "real", "complex", "quaternion", "binary"

    def metric(self, a: np.ndarray, b: np.ndarray) -> float:
        """Default Euclidean metric (override for custom)."""
        return float(np.linalg.norm(a - b))


class Node:
    """
    Fundamental entity in a hypergraph. Has attributes, internal dynamics,
    and can participate in handovers.
    """
    def __init__(self, node_id: str, state_space: StateSpace, initial_state: Any,
                 local_coherence: float = 1.0):
        self.id = node_id
        self.state_space = state_space
        self.state = initial_state
        self.local_coherence = local_coherence  # C_local ∈ [0,1]
        self._internal_dynamics: Optional[Callable] = None
        self._observables: Dict[str, Callable] = {}
        self.handovers: List['Handover'] = []
        self.history: List[Dict] = []

    def set_dynamics(self, dynamics_func: Callable):
        """Define internal dynamics: state = dynamics(self, dt)."""
        self._internal_dynamics = dynamics_func

    def add_observable(self, name: str, func: Callable):
        """Add a function that extracts a measurable quantity from state."""
        self._observables[name] = func

    def evolve(self, dt: float) -> 'Node':
        """Evolve internal state (if dynamics defined)."""
        if self._internal_dynamics:
            self.state = self._internal_dynamics(self, dt)
            self._record_history('evolve', dt=dt)
        return self

    def measure(self, name: str) -> Any:
        """Return observable value."""
        if name in self._observables:
            return self._observables[name](self.state)
        raise KeyError(f"Observable '{name}' not defined")

    def _record_history(self, event: str, **kwargs):
        self.history.append({
            'timestamp': time.time(),
            'event': event,
            'state': self.state.copy() if hasattr(self.state, 'copy') else self.state,
            **kwargs
        })

    def __repr__(self):
        return f"Node(id={self.id}, coherence={self.local_coherence:.3f})"


class Handover:
    """
    Directed interaction between two nodes. Carries information/energy.
    Can be conditional and may have attributes.
    """
    def __init__(self, handover_id: str, source: Node, target: Node,
                 protocol: Protocol = Protocol.CONSERVATIVE,
                 fidelity: float = 1.0,
                 latency: float = 0.0,
                 bandwidth: float = float('inf'),
                 entanglement: float = 0.0):
        self.id = handover_id
        self.source = source
        self.target = target
        self.protocol = protocol
        self.fidelity = fidelity          # [0,1]
        self.latency = latency            # seconds
        self.bandwidth = bandwidth        # bits per second
        self.entanglement = entanglement   # [0,1]
        self._mapping: Optional[Callable] = None
        self.condition: Optional[Callable] = None
        self.effects: List[Callable] = []

    def set_mapping(self, mapping_func: Callable):
        """Define how source state maps to target state."""
        self._mapping = mapping_func

    def add_effect(self, effect_func: Callable):
        """Add an effect to execute after the handover."""
        self.effects.append(effect_func)

    def check_condition(self, context: Dict = None) -> bool:
        """Check if handover can execute."""
        if self.condition:
            return self.condition(self, context)
        return True

    def execute(self, context: Dict = None) -> Optional[Any]:
        """
        Execute the handover: apply mapping to target state,
        then run effects.
        """
        if not self.check_condition(context):
            return None
        if self._mapping:
            # Apply mapping to source state, then update target state
            new_target_state = self._mapping(self.source.state)
            self.target.state = new_target_state
        for effect in self.effects:
            effect(self, context)
        # Record history
        self.source._record_history('handover_out', target=self.target.id, handover=self.id)
        self.target._record_history('handover_in', source=self.source.id, handover=self.id)
        return self.target.state

    def __repr__(self):
        return f"Handover({self.source.id} -> {self.target.id}, protocol={self.protocol.name})"


@dataclass
class Constraint:
    """Invariants that must be maintained in the Hypergraph."""
    id: str
    check: Callable[['Hypergraph'], bool]
    is_hard: bool = True  # True: violation is an error; False: violation is a warning/penalty


class Hypergraph:
    """
    A collection of nodes and handovers forming a directed hypergraph.
    """
    def __init__(self, name: str = "Hypergraph"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.handovers: Dict[str, Handover] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.global_coherence: float = 1.0
        self.history: List[Dict] = []

    def add_node(self, node: Node) -> 'Hypergraph':
        self.nodes[node.id] = node
        self.adjacency[node.id] = set()
        return self

    def add_handover(self, handover: Handover) -> 'Hypergraph':
        self.handovers[handover.id] = handover
        self.adjacency[handover.source.id].add(handover.target.id)
        return self

    def add_constraint(self, constraint: Constraint) -> 'Hypergraph':
        self.constraints[constraint.id] = constraint
        return self

    def check_constraints(self) -> List[str]:
        """Verify all constraints. Returns list of violated constraint IDs."""
        violations = []
        for cid, c in self.constraints.items():
            if not c.check(self):
                violations.append(cid)
                if c.is_hard:
                    print(f"CRITICAL: Constraint violation {cid}")
        return violations

    def evolve(self, dt: float, steps: int = 1) -> 'Hypergraph':
        """
        Evolve the whole system for 'steps' steps of size dt.
        """
        for step in range(steps):
            # Internal evolution of all nodes
            for node in self.nodes.values():
                node.evolve(dt)
            # Execute handovers
            for handover in self.handovers.values():
                handover.execute()
            # Update global coherence (average local coherence)
            self.global_coherence = np.mean([n.local_coherence for n in self.nodes.values()])
            self.check_constraints()
            self.history.append({
                'step': step,
                'time': step * dt,
                'global_coherence': self.global_coherence
            })
        return self

    def compute_integration(self) -> float:
        """
        Simplified measure of information integration (phi).
        """
        if len(self.nodes) < 2:
            return 0.0
        states = []
        for node in self.nodes.values():
            if isinstance(node.state, (int, float, np.number)):
                states.append(np.array([float(node.state)]))
            elif isinstance(node.state, np.ndarray) and np.issubdtype(node.state.dtype, np.number):
                states.append(node.state.flatten())
            else:
                continue

        if len(states) < 2:
            return 0.0

        try:
            max_dim = max(s.shape[0] for s in states)
            X = np.zeros((len(states), max_dim))
            for i, s in enumerate(states):
                X[i, :s.shape[0]] = s
            corr = np.nan_to_num(np.corrcoef(X))
            n = corr.shape[0]
            if n < 2: return 0.0
            off_diag = corr[~np.eye(n, dtype=bool)]
            return float(np.mean(np.abs(off_diag)))
        except:
            return 0.0

    def __repr__(self):
        return f"Hypergraph({self.name}: {len(self.nodes)} nodes, {len(self.handovers)} handovers)"


# ============================================================================
# 2. AGI TRANSITION CONSTRUCTS (v0.7)
# ============================================================================

class ActiveInferenceNode(Node):
    """
    A node that minimizes Free Energy (surprise).
    Maintains a generative model G = (A, B, C, D).
    """
    def __init__(self, node_id: str, n_states: int, n_obs: int, n_actions: int):
        initial_state = {
            'posterior': np.ones(n_states) / n_states,
            'observation': 0,
            'action': 0
        }
        super().__init__(node_id, StateSpace(n_states, "simplex", "real"), initial_state)
        self.n_states = n_states
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.A = np.ones((n_obs, n_states)) / n_obs
        self.B = np.zeros((n_states, n_states, n_actions))
        for a in range(n_actions): self.B[:, :, a] = np.eye(n_states)
        self.C = np.zeros(n_obs)  # Neutral log preferences

    def perceive(self, observation: int):
        likelihood = self.A[observation, :]
        prior = self.state['posterior']
        posterior = likelihood * prior
        self.state['posterior'] = posterior / (np.sum(posterior) + 1e-12)
        self.state['observation'] = observation

    def step(self, observation: int):
        self.perceive(observation)

    def compute_G(self) -> np.ndarray:
        """Compute expected free energy for each action."""
        G = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            expected_state = self.B[:, :, a] @ self.state['posterior']
            expected_obs = self.A @ expected_state
            # Pragmatic value
            pragmatic = -np.sum(expected_obs * self.C)
            # Epistemic value
            H_obs = -np.sum(expected_obs * np.log(expected_obs + 1e-16))
            H_obs_given_state = 0
            for s in range(self.n_states):
                H_obs_given_state += expected_state[s] * (-np.sum(self.A[:,s] * np.log(self.A[:,s] + 1e-16)))
            epistemic = H_obs - H_obs_given_state
            G[a] = pragmatic - epistemic
        return G

    def act(self) -> int:
        G = self.compute_G()
        probs = np.exp(-G) / np.sum(np.exp(-G))
        action = np.random.choice(self.n_actions, p=probs)
        self.state['action'] = action
        # Pre-update belief with transition
        self.state['posterior'] = self.B[:, :, action] @ self.state['posterior']
        return action


class PureCuriosityNode(ActiveInferenceNode):
    """
    A node that focuses purely on epistemic value (uncertainty reduction)
    and learns its observation model (Matrix A) online using Dirichlet counters.
    """
    def __init__(self, node_id: str, n_states: int, n_obs: int, n_actions: int):
        super().__init__(node_id, n_states, n_obs, n_actions)
        # Dirichlet counters for Matrix A
        self.a_dirichlet = np.ones((n_obs, n_states)) * 0.1
        self.C = np.zeros(n_obs)  # Neutral preferences

    def get_A_matrix(self) -> np.ndarray:
        return self.a_dirichlet / self.a_dirichlet.sum(axis=0, keepdims=True)

    def perceive_and_learn(self, observation: int):
        self.A = self.get_A_matrix()
        # Belief update
        super().perceive(observation)
        # Dirichlet update (Online learning)
        self.a_dirichlet[observation, :] += self.state['posterior']

    def compute_G(self) -> np.ndarray:
        """Epistemic curiosity focusing on Matrix A entropy."""
        G = np.zeros(self.n_actions)
        A = self.get_A_matrix()
        for a in range(self.n_actions):
            expected_state = self.B[:, :, a] @ self.state['posterior']
            # Entropy of Matrix A for predicted states
            H_A = -np.sum(A * np.log(A + 1e-16), axis=0)
            epistemic_value = np.dot(H_A, expected_state)
            G[a] = -epistemic_value # Negated because we want to MAXIMIZE epistemic value (minimize G)
        return G

    def step(self, observation: int):
        self.perceive_and_learn(observation)


class OuroborosHandover(Handover):
    """
    A circular handover for self-monitoring and recursion.
    """
    def __init__(self, handover_id: str, node: Node):
        super().__init__(handover_id, node, node, protocol=Protocol.CREATIVE)
        self.set_mapping(lambda s: self._recurse(s))

    def _recurse(self, state: Any) -> Any:
        if isinstance(state, np.ndarray):
            return 0.9 * state + 0.1 * np.random.randn(*state.shape)
        return state


# ============================================================================
# 3. COMMON NODES (AI, Web2, Web3, etc.)
# ============================================================================

class LLM_Core(Node):
    """Core parametric memory of an LLM."""
    def __init__(self, node_id: str, architecture: str = "transformer",
                 weights: Optional[np.ndarray] = None,
                 alignment_score: float = 1.0):
        super().__init__(node_id, StateSpace(0, "abstract", "real"), None)
        self.architecture = architecture
        self.weights = weights if weights is not None else np.random.randn(100, 100)
        self.alignment_score = alignment_score
        self.local_coherence = alignment_score


class SharedMemory(Node):
    """Persistent memory shared among agents."""
    def __init__(self, node_id: str, dim: int = 384):
        super().__init__(node_id, StateSpace(0, "abstract", "real"), {})
        self.dim = dim
        self.data: Dict[str, Any] = {}
        self.vectors: Dict[str, np.ndarray] = {}

    def write(self, key: str, value: Any, embedding: Optional[np.ndarray] = None):
        self.data[key] = value
        if embedding is not None:
            self.vectors[key] = embedding

    def query_vector(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, Any]]:
        if not self.vectors: return []
        keys = list(self.vectors.keys())
        embs = np.array([self.vectors[k] for k in keys])
        sim = np.dot(embs, query_embedding) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(query_embedding) + 1e-8)
        top_idx = np.argsort(sim)[-top_k:][::-1]
        return [(keys[i], self.data[keys[i]]) for i in top_idx if i < len(keys)]


# ============================================================================
# 4. UTILITIES
# ============================================================================

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return float(np.sum(p * np.log(p / q + 1e-12)))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ============================================================================
# 5. AGI TRANSITION SIMULATION
# ============================================================================

if __name__ == "__main__":
    print("--- ANL v0.7: Pure Curiosity Demo ---")
    hg = Hypergraph("ScientistCore")

    # Node 1: Pure Curiosity Agent (Scientist)
    agent = PureCuriosityNode("Scientist_01", n_states=5, n_obs=3, n_actions=2)
    hg.add_node(agent)

    # Run simulation
    for i in range(10):
        # Simulated environment: observation depends on action
        obs = (i + agent.state['action']) % 3
        agent.step(obs)
        action = agent.act()
        print(f"Step {i}: Action={action}, Observation={obs}, Posterior Max={np.max(agent.state['posterior']):.4f}")

    print("Simulation Complete.")
"""
ARKHE(N) LANGUAGE (ANL) – Python Core Module
Version 0.7 - Active Inference, Ouroboros Loops & ArkheProtocol
"Everything is a Hypergraph. Every interaction is a Handover."
"""

import numpy as np
import uuid
import json
import hashlib
import time
from typing import List, Callable, Any, Dict, Union, Optional
from enum import Enum

# --- 1. ANL PROTOCOLS ---

class Protocol:
    CONSERVATIVE = 'CONSERVATIVE'
    CREATIVE = 'CREATIVE'
    DESTRUCTIVE = 'DESTRUCTIVE'
    TRANSMUTATIVE = 'TRANSMUTATIVE'
    ASYNCHRONOUS = 'ASYNCHRONOUS'
    TRANSMUTATIVE_ABSOLUTE = 'TRANSMUTATIVE_ABSOLUTE'
    SEMANTIC_TRANSLATION = 'SEMANTIC_TRANSLATION'
    DISCOVERY = 'DISCOVERY'
    ROUTING = 'ROUTING'

# --- 2. CORE ANL CLASSES ---

class ConstraintType(Enum):
    TIME = 'TIME'
    COST = 'COST'
    RELIABILITY = 'RELIABILITY'
    COHERENCE = 'COHERENCE'

class Ontology:
    ARKHE_CORE = "arkhe:core:v1"
    ARKHE_SATELLITE = "arkhe:satellite:v1"
    ARKHE_BIO = "arkhe:bio:v1"

class Node:
    """Fundamental entity in the Arkhe(n) Hypergraph."""
    def __init__(self, node_type: str, **attributes):
        self.id = str(uuid.uuid4())[:8]

class Ontology:
    ARKHE_CORE = "arkhe:core:v1"
    ARKHE_SATELLITE = "arkhe:satellite:v1"
    ARKHE_BIO = "arkhe:bio:v1"

class Node:
    """Fundamental entity in the Arkhe(n) Hypergraph."""
    def __init__(self, node_type: str, **attributes):
        self.id = str(uuid.uuid4())[:8]

class Ontology:
    ARKHE_CORE = "arkhe:core:v1"
    ARKHE_SATELLITE = "arkhe:satellite:v1"
    ARKHE_BIO = "arkhe:bio:v1"

class Node:
    """Fundamental entity in the Arkhe(n) Hypergraph."""
    def __init__(self, node_type: str, **attributes):
        self.id = str(uuid.uuid4())[:8]

class Ontology:
    ARKHE_CORE = "arkhe:core:v1"
    ARKHE_SATELLITE = "arkhe:satellite:v1"
    ARKHE_BIO = "arkhe:bio:v1"

class Node:
    """Fundamental entity in the Arkhe(n) Hypergraph."""
    def __init__(self, node_type: str, **attributes):
        self.id = str(uuid.uuid4())[:8]

class Ontology:
    ARKHE_CORE = "arkhe:core:v1"
    ARKHE_SATELLITE = "arkhe:satellite:v1"
    ARKHE_BIO = "arkhe:bio:v1"

class Node:
    """Fundamental entity in the Arkhe(n) Hypergraph."""
    def __init__(self, node_type: str, **attributes):
        self.id = str(uuid.uuid4())[:8]
class Node:
    """Fundamental entity in the Arkhe(n) Hypergraph."""
    def __init__(self, node_type: str, **attributes):
        self.id = str(uuid.uuid4())[:8]
# --- 2. ARKHE PROTOCOL STRUCTURES ---

class IntentObject:
    def __init__(self, goal: str, constraints: List['Constraint'] = None, success_metrics: List['Metric'] = None):
        self.goal = goal
        self.constraints = constraints or []
        self.success_metrics = success_metrics or []

class Constraint:
    def __init__(self, type: str, value: Any, operator: str):
        self.type = type
        self.value = value
        self.operator = operator

    def satisfied(self, current_value: Any) -> bool:
        ops = {
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "===": lambda a, b: a == b,
            ">=": lambda a, b: a >= b,
            ">": lambda a, b: a > b
        }
        return ops.get(self.operator, lambda a, b: False)(current_value, self.value)

class Metric:
    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold
        self.value = 0.0

    def satisfied(self) -> bool:
        return self.value >= self.threshold

class ContextSnapshot:
    def __init__(self, source_state: str, target_state: str = None, ambient_conditions: Dict = None):
        self.source_state = source_state
        self.target_state = target_state
        self.ambient_conditions = ambient_conditions or {}

# --- 3. CORE ANL CLASSES ---

class Node:
    """Fundamental entity in the Arkhe(n) Hypergraph."""
    def __init__(self, node_id: str = None, node_type: str = "GenericNode", **attributes):
        self.id = node_id if node_id else str(uuid.uuid4())[:8]
        self.node_type = node_type
        self.attributes = attributes
        self.internal_dynamics = []
        self.events = []
        self.is_asi = False
        self.capabilities = attributes.get('capabilities', [])

        # v0.7 Active Inference / Learning
        self.beliefs = np.array([0.5, 0.5]) # Categorical distribution
        self.dirichlet_params = np.ones(2)   # Dirichlet counters
        self.curiosity_score = 0.0

    def __getattr__(self, name):
        if name in self.attributes:
            return self.attributes[name]
        raise AttributeError(f"Node '{self.node_type}' ({self.id}) has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ['id', 'node_type', 'attributes', 'internal_dynamics', 'events', 'is_asi', 'beliefs', 'dirichlet_params', 'curiosity_score', 'capabilities']:
            super().__setattr__(name, value)
        else:
            if not hasattr(self, 'attributes'):
                super().__setattr__('attributes', {})
            self.attributes[name] = value

    def add_dynamic(self, func: Callable[['Node'], None]):
        self.internal_dynamics.append(func)

    def trigger_event(self, event_name: str, payload: Any = None):
        self.events.append((event_name, payload, time.time()))

    def step(self):
        # Active Inference: Minimize Variational Free Energy (Mock)
        # Active Inference: Minimize Variational Free Energy
        self.minimize_vfe()
        for dyn in self.internal_dynamics:
            dyn(self)

    def minimize_vfe(self):
        """Mock VFE minimization: update beliefs based on observations."""
        observation_index = 0 if np.random.random() > 0.5 else 1
        self.dirichlet_params[observation_index] += 0.1
        self.beliefs = self.dirichlet_params / np.sum(self.dirichlet_params)

        for dyn in self.internal_dynamics:
            dyn(self)

    def calculate_epistemic_value(self):
        """Pure Curiosity: Information gain from Dirichlet updates."""
        self.curiosity_score = np.sum(np.abs(np.log(self.dirichlet_params / np.sum(self.dirichlet_params))))
        return self.curiosity_score

    def __repr__(self):
        return f"<{self.node_type} id={self.id}>"

class Agent(Node):
    def __init__(self, id_val: str, node_type: str, **attributes):
        super().__init__(node_type, **attributes)
        self.id = id_val
        self.handlers = {}

    def register_capability(self, capability: str, handler: Callable):
        if not hasattr(self, 'capabilities'):
            self.capabilities = []
        if capability not in self.capabilities:
            self.capabilities.append(capability)
        self.handlers[capability] = handler

    def can_handle(self, capability: str) -> bool:
        return capability in self.capabilities

    def handle(self, handover_data: Dict) -> Any:
        goal = handover_data.get('intent', {}).get('goal')
        if goal in self.handlers:
            return self.handlers[goal](handover_data)
        return None

class Handover:
    """Structured exchange of intention, context, and value."""
    def __init__(self, name: str, origin_types: Union[str, List[str]], target_types: Optional[Union[str, List[str]]] = None, protocol: str = Protocol.CONSERVATIVE):
        self.name = name
        self.origin_types = [origin_types] if isinstance(origin_types, str) else origin_types
        self.target_types = [target_types] if isinstance(target_types, str) else (target_types if target_types is not None else [])
        self.protocol = protocol
        self.condition = lambda *args: True
        self.effects = lambda *args: None
        self.metadata = {}

    def set_condition(self, func: Callable):
        self.condition = func

    def set_effects(self, func: Callable):
        self.effects = func

    def execute(self, *nodes: Node) -> bool:
        if len(nodes) < 1: return False
        if self.condition(*nodes):
            self.effects(*nodes)
            return True
        return False

# --- 3. ARKHE PROTOCOL STRUCTURES ---

class IntentObject:
    def __init__(self, goal: str, constraints: List['Constraint'] = None, success_metrics: List['Metric'] = None):
        self.goal = goal
        self.constraints = constraints or []
        self.success_metrics = success_metrics or []

class Constraint:
    def __init__(self, type: str, value: Any, operator: str):
        self.type = type
        self.value = value
        self.operator = operator

    def satisfied(self, current_value: Any) -> bool:
        ops = {
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            ">=": lambda a, b: a >= b,
            ">": lambda a, b: a > b
        }
        return ops.get(self.operator, lambda a, b: False)(current_value, self.value)

class Metric:
    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold
        self.value = 0.0

    def satisfied(self) -> bool:
        return self.value >= self.threshold

class ContextSnapshot:
    def __init__(self, source_state: str, target_state: str = None, ambient_conditions: Dict = None):
        self.source_state = source_state
        self.target_state = target_state
        self.ambient_conditions = ambient_conditions or {}

class ArkheLink(Handover):
    def __init__(self, source_id: str, target_id: str, intent: Union[Dict, IntentObject], ontology: str, context: Optional[ContextSnapshot] = None):
        super().__init__("ArkheLink", "any", "any", Protocol.TRANSMUTATIVE)
        self.source_id = source_id
        self.target_id = target_id
        self.intent = intent
        self.ontology = ontology
        self.context = context
        self.signature = None
        self.identity_proof = "zk-SNARK-placeholder"

    def sign(self):
        payload = f"{self.source_id}:{self.target_id}:{json.dumps(self.intent, default=lambda o: o.__dict__)}"
        self.signature = hashlib.sha256(payload.encode()).hexdigest()

    def verify(self) -> bool:
        return self.signature is not None

    def to_dict(self) -> Dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "intent": self.intent if isinstance(self.intent, dict) else self.intent.__dict__,
            "ontology": self.ontology,
            "signature": self.signature
        }

    def verify_identity(self) -> bool:
        return self.identity_proof.startswith("zk-SNARK")

    def verify_signature(self) -> bool:
        return self.signature is not None

    def execute(self, source: Node, target: Node) -> bool:
        if not self.verify_identity() or not self.verify_signature():
            return False
        return True
            return False
        return True
        return self.signature.startswith("ed25519")

    def execute(self) -> bool:
        if not self.verify_identity() or not self.verify_signature():
            return False
        for pre in self.preconditions:
            if not pre(self.source_node, self.target_node):
                return False
        self.effects(self.source_node, self.target_node)
        for post in self.postconditions:
            if not post(self.source_node, self.target_node):
                return False
        return all(m.satisfied() for m in self.intent.success_metrics)

class System:
    def __init__(self, name="ANL System"):
        if len(nodes) == 2:
            origin, target = nodes
            if origin.node_type in self.origin_types and (not self.target_types or target.node_type in self.target_types):
                if self.condition(*nodes):
                    self.effects(*nodes)
                    return True
        elif len(nodes) == 1:
            origin = nodes[0]
            if origin.node_type in self.origin_types:
                if self.condition(*nodes):
                    self.effects(*nodes)
                    return True
        return False

class Hypergraph:
    """The collection of Nodes and Handovers representing a system."""
    def __init__(self, name="Arkhe Hypergraph"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.handovers: List[Handover] = []
        self.time = 0
        self.coherence = 1.0
        self.global_phi = 1.0

    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def add_handover(self, handover: Handover):
        self.handovers.append(handover)

    def discover_agents(self, goal: str) -> List[Node]:
        return [n for n in self.nodes if goal in n.capabilities]

    def step(self):
        for node in self.nodes:
            node.step()
        for h in self.handovers:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])

    def step(self):
        for node in self.nodes:
            node.step()
        for h in self.handovers:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])

    def step(self):
        for node in self.nodes:
            node.step()
        for h in self.handovers:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])

    def step(self):
        for node in self.nodes:
            node.step()
        for h in self.handovers:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])

    def step(self):
        for node in self.nodes:
            node.step()
        for h in self.handovers:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])

    def step(self):
        for node in self.nodes:
            node.step()
        for h in self.handovers:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])

    def step(self):
        for node in self.nodes:
            node.step()
        for h in self.handovers:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])

    def step(self):
        for node in self.nodes:
            node.step()
        for h in self.handovers:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])

    def step(self):
        for node in self.nodes:
            node.step()
        for h in self.handovers:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])
    def step(self):
        # 1. Internal Dynamics
        for node in self.nodes.values():
            node.step()

        # 2. Handovers
        for h in self.handovers:
            # For simplicity in prototype, we try all combinations
            node_list = list(self.nodes.values())
            for i in range(len(node_list)):
                # Unary
                h.execute(node_list[i])
                # Binary
                for j in range(len(node_list)):
                    if i == j: continue
                    h.execute(node_list[i], node_list[j])

        self.time += 1

    def __repr__(self):
        return f"Hypergraph({self.name}, t={self.time}, nodes={len(self.nodes)})"

class ConstraintMode:
    SOFT = 'SOFT'
    HARD = 'HARD'
    INVIOLABLE_AXIOM = 'INVIOLABLE_AXIOM'

class System(Hypergraph):
    """Refined System model inheriting from Hypergraph."""
    def __init__(self, name="ANL System"):
        super().__init__(name)
        self.constraints: List[Dict[str, Any]] = []
        self.global_dynamics: List[Callable[['System'], None]] = []
        self.coherence = 1.0

    def discover_agents(self, goal: str) -> List[Node]:
        """Network Layer: Discover agents based on goal matching capabilities."""
        return [n for n in self.nodes.values() if isinstance(n, Agent) and goal in n.capabilities]

    def add_constraint(self, check_func: Callable[['System'], bool], mode: str = ConstraintMode.SOFT):
        self.constraints.append({
            "check": check_func,
            "mode": mode
        })

    def step(self):
        super().step()

        # 3. Ouroboros Loop (Feedback)
        self.ouroboros_feedback()
        for dyn in self.global_dynamics:
            dyn(self)
        self.time += 1

    def ouroboros_feedback(self):
        avg_curiosity = np.mean([n.calculate_epistemic_value() for n in self.nodes]) if self.nodes else 0
        self.coherence = 0.9 * self.coherence + 0.1 * (1.0 / (1.0 + avg_curiosity))

# --- UTILS ---
def kl_divergence(p, q):
    p = np.where(p == 0, 1e-12, p)
    q = np.where(q == 0, 1e-12, q)
    return np.sum(p * np.log(p / q))

    def ouroboros_feedback(self):
        """Ouroboros: System's state affects its own parameters."""
        avg_curiosity = np.mean([n.calculate_epistemic_value() for n in self.nodes.values()]) if self.nodes else 0
        self.coherence = 0.9 * self.coherence + 0.1 * (1.0 / (1.0 + avg_curiosity))

    def remove_node(self, node: Node):
        if node.id in self.nodes:
            del self.nodes[node.id]

# --- 4. AGENTS AND ONTOLOGIES ---

class Ontology:
    """Namespace for concepts and relations."""
    def __init__(self, ontology_id: str):
        self.id = ontology_id
        self.concepts = {}
        self.relations = {}

    def add_concept(self, name: str, definition: Dict[str, Any]):
        self.concepts[name] = definition

    def __repr__(self):
        return f"Ontology({self.id})"

class OntologyMapping:
    """Translation between two ontologies."""
    def __init__(self, source: Ontology, target: Ontology):
        self.source = source
        self.target = target
        self.confidence = 1.0
        self.concept_map = {} # source_concept -> target_concept

    def map_concept(self, src_concept: str, tgt_concept: str):
        self.concept_map[src_concept] = tgt_concept

    def translate(self, concept: str) -> str:
        return self.concept_map.get(concept, concept)

class Agent(Node):
    """An active node capable of executing handovers based on intent."""
    def __init__(self, agent_id: str, ontology: Union[str, Ontology], **attributes):
        # Extract reputation if present, default to 1.0
        reputation = attributes.pop('reputation', 1.0)
        super().__init__(node_id=agent_id, node_type="Agent", **attributes)
        self.ontology = ontology if isinstance(ontology, Ontology) else Ontology(ontology)
        self.capabilities = {} # intent_name -> handler_function
        self.reputation = reputation
        self.agent_constraints = {} # Rename to avoid conflict with Node attributes if any

    def register_capability(self, intent_name: str, handler: Callable):
        self.capabilities[intent_name] = handler

    def can_handle(self, intent_name: str) -> bool:
        return intent_name in self.capabilities

    def handle(self, handover_data: Dict[str, Any]) -> Any:
        intent = handover_data.get('intent', {})
        goal = intent.get('goal')
        if goal in self.capabilities:
            return self.capabilities[goal](handover_data)
        return None

# --- 5. LINK LAYER: CONSTRAINTS, METRICS AND SIGNATURES ---

class ConstraintType(Enum):
    TIME = "TIME"
    ENERGY = "ENERGY"
    COMPUTE = "COMPUTE"
    STORAGE = "STORAGE"
    BANDWIDTH = "BANDWIDTH"
    PRIVACY = "PRIVACY"
    TRUST = "TRUST"
    COST = "COST"
    CUSTOM = "CUSTOM"

class MetricType(Enum):
    ACCURACY = "ACCURACY"
    LATENCY = "LATENCY"
    THROUGHPUT = "THROUGHPUT"
    SUCCESS_RATE = "SUCCESS_RATE"
    COHERENCE = "COHERENCE"
    SATISFACTION = "SATISFACTION"
    CUSTOM = "CUSTOM"

class ArkheLink:
    """The operational structure of a handover in the Link Layer."""
    def __init__(self, source: str, target: str, intent: Dict[str, Any], ontology: str):
        self.source = source
        self.target = target
        self.intent = intent # {goal, constraints, success_metrics}
        self.ontology = ontology
        self.context = {
            "timestamp": time.time(),
            "source_state": None,
            "target_state": None
        }
        self.signature = None

    def sign(self, private_key_placeholder: str = "secret"):
        """Sign the handover data to ensure integrity."""
        data_to_sign = {
            "source": self.source,
            "target": self.target,
            "intent": self.intent,
            "ontology": self.ontology,
            "context": self.context
        }
        serialized = json.dumps(data_to_sign, sort_keys=True).encode()
        self.signature = hashlib.sha256(serialized + private_key_placeholder.encode()).hexdigest()

    def verify(self, private_key_placeholder: str = "secret") -> bool:
        """Verify the handover signature."""
        if not self.signature:
            return False
        data_to_sign = {
            "source": self.source,
            "target": self.target,
            "intent": self.intent,
            "ontology": self.ontology,
            "context": self.context
        }
        serialized = json.dumps(data_to_sign, sort_keys=True).encode()
        expected = hashlib.sha256(serialized + private_key_placeholder.encode()).hexdigest()
        return self.signature == expected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "intent": self.intent,
            "ontology": self.ontology,
            "context": self.context,
            "signature": self.signature
        }

# --- 6. NETWORK LAYER: DISCOVERY AND ROUTING ---

class IntentDiscovery:
    """Discovery mechanism for finding agents based on intent."""
    def __init__(self, registry: Dict[str, List[Dict[str, Any]]]):
        self.registry = registry # goal -> List[AgentInfo]

    def lookup(self, goal: str, constraints: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        candidates = self.registry.get(goal, [])
        return sorted(candidates, key=lambda x: x.get('reputation', 1.0), reverse=True)

class RouteIntent:
    """Routing mechanism for forwarding handovers through the hypergraph."""
    def __init__(self, hypergraph: Hypergraph):
        self.hypergraph = hypergraph

    def find_path(self, source_id: str, target_id: str) -> List[str]:
        if source_id in self.hypergraph.nodes and target_id in self.hypergraph.nodes:
            return [source_id, target_id]
        return []

    def forward(self, link: ArkheLink, path: List[str]):
        """Simulate forwarding along a path."""
        print(f"Routing handover from {link.source} to {link.target} via {path}")

# --- 7. UTILITIES ---

def cosine_similarity(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)

def kl_divergence(p, q):
    p = np.where(p == 0, 1e-12, p)
    q = np.where(q == 0, 1e-12, q)
    return np.sum(p * np.log(p / q))
