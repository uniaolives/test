#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARKHE(N) LANGUAGE (ANL) – Core Python Module (v0.7)
===================================================
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
