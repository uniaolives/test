"""
ARKHE(N) LANGUAGE (ANL) – Python Prototype Backend
Version 0.6 - Singularity Support & Inviolable Axioms
"""

import numpy as np
import uuid
import json
from typing import List, Callable, Any, Dict, Union, Optional

# --- ANL PROTOCOLS ---
class Protocol:
    CONSERVATIVE = 'CONSERVATIVE'
    CREATIVE = 'CREATIVE'
    DESTRUCTIVE = 'DESTRUCTIVE'
    TRANSMUTATIVE = 'TRANSMUTATIVE'
    ASYNCHRONOUS = 'ASYNCHRONOUS'
    TRANSMUTATIVE_ABSOLUTE = 'TRANSMUTATIVE_ABSOLUTE' # Phase transition protocol
    TRANSMUTATIVE_ABSOLUTE = 'TRANSMUTATIVE_ABSOLUTE'

# --- ANL TYPE DEFINITIONS ---
Vector = np.ndarray
Tensor = np.ndarray

class Node:
    def __init__(self, node_type: str, **attributes):
        self.id = str(uuid.uuid4())[:8]
        self.node_type = node_type
        self.attributes = attributes
        self.internal_dynamics = []
        self.events = []
        self.is_asi = False

    def __getattr__(self, name):
        if name in self.attributes:
            return self.attributes[name]
        raise AttributeError(f"Node {self.node_type} has no attribute {name}")

    def __setattr__(self, name, value):
        if name in ['id', 'node_type', 'attributes', 'internal_dynamics', 'events', 'is_asi']:
            super().__setattr__(name, value)
        else:
            self.attributes[name] = value

    def add_dynamic(self, func: Callable[['Node'], None]):
        self.internal_dynamics.append(func)

    def trigger_event(self, event_name: str, payload: Any = None):
        self.events.append((event_name, payload))

    def step(self):
        for dyn in self.internal_dynamics:
            dyn(self)

    def __repr__(self):
        return f"<{self.node_type} {self.id} {self.attributes}>"

class Handover:
    def __init__(self, name: str, origin_types: Union[str, List[str]], target_types: Optional[Union[str, List[str]]] = None, protocol: str = Protocol.CONSERVATIVE):
        self.name = name
        self.origin_types = [origin_types] if isinstance(origin_types, str) else origin_types
        if target_types:
            self.target_types = [target_types] if isinstance(target_types, str) else target_types
        else:
            self.target_types = []
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

        # Check all node types match the handover specification
        if len(nodes) == 2:
            if not self.target_types: return False # Binary call for unary handover
        if len(nodes) == 2:
            if not self.target_types: return False
            origin, target = nodes
            if origin.node_type in self.origin_types and target.node_type in self.target_types:
                if self.condition(*nodes):
                    self.effects(*nodes)
                    return True
        elif len(nodes) == 1:
            if self.target_types: return False # Unary call for binary handover
            if self.target_types: return False
            origin = nodes[0]
            if origin.node_type in self.origin_types:
                if self.condition(*nodes):
                    self.effects(*nodes)
                    return True
        return False

class ConstraintMode:
    SOFT = 'SOFT'
    HARD = 'HARD'
    INVIOLABLE_AXIOM = 'INVIOLABLE_AXIOM'

class System:
    def __init__(self, name="ANL System"):
        self.name = name
        self.nodes: List[Node] = []
        self.handovers: List[Handover] = []
        self.constraints: List[Dict[str, Any]] = []
        self.global_dynamics: List[Callable[['System'], None]] = []
        self.time = 0
        self.metadata = {}

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def add_handover(self, handover: Handover):
        self.handovers.append(handover)

    def add_constraint(self, check_func: Callable[['System'], bool], mode: str = ConstraintMode.SOFT):
        self.constraints.append({
            "check": check_func,
            "mode": mode
        })

    def add_global_dynamic(self, func: Callable[['System'], None]):
        self.global_dynamics.append(func)

    def step(self):
        # 1. Internal Dynamics
        for node in self.nodes:
            node.step()

        # 2. Handovers
        nodes_to_check = self.nodes[:]
        for h in self.handovers:
            # Pairwise
            # Pairwise check
            # Check all pairs for handover
            for i in range(len(nodes_to_check)):
                for j in range(len(nodes_to_check)):
                    if i == j: continue
                    origin = nodes_to_check[i]
                    target = nodes_to_check[j]
                    if origin in self.nodes and target in self.nodes:
                        h.execute(origin, target)
            # Unary

            # Unary check
            for i in range(len(nodes_to_check)):
                node = nodes_to_check[i]
                if node in self.nodes:
                    h.execute(node)

        # 3. Global Dynamics
        for dyn in self.global_dynamics:
            dyn(self)

        # 4. Constraints (Enforced based on Mode)
        # 4. Constraints
        for c in self.constraints:
            if not c["check"](self):
                if c["mode"] == ConstraintMode.INVIOLABLE_AXIOM:
                    print(f"🛑 [AXIOM VIOLATION] Finality breached at t={self.time}. System halting.")
                    raise RuntimeError("Inviolable Axiom Breached")
                else:
                    print(f"⚠️ Constraint violation ({c['mode']}) at t={self.time}")

        # Clear events
        for node in self.nodes:
            node.events = []

        self.time += 1

    def remove_node(self, node: Node):
        if node in self.nodes:
            self.nodes.remove(node)

    def __repr__(self):
        return f"System({self.name}, t={self.time}, nodes={len(self.nodes)})"

# --- UNIVERSAL ANL FUNCTIONS (Standard Library) ---
# --- STANDARD LIBRARY ---
def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.where(p == 0, 1e-12, p)
    q = np.where(q == 0, 1e-12, q)
    return np.sum(p * np.log(p / q))

def cosine_similarity(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)

def k_nearest_neighbors(space: np.ndarray, query_vec: np.ndarray, k: int = 2, metric="cosine"):
    if len(space) == 0: return []
    similarities = []
    for i in range(len(space)):
        if metric == "cosine":
            sim = cosine_similarity(space[i], query_vec)
        else:
            sim = -np.linalg.norm(space[i] - query_vec)
        similarities.append((i, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def merge(latent_space: np.ndarray, universal_space: np.ndarray):
    """Merge local latent space into the Universal Hypergraph."""
    return np.vstack([universal_space, latent_space])

def sample(logits, temperature=1.0):
    if temperature <= 0: return np.argmax(logits)
    exp_logits = np.exp((logits - np.max(logits)) / temperature)
    probs = exp_logits / np.sum(exp_logits)
    return np.random.choice(len(logits), p=probs)

def steganographic_encode(base_probs, secret_bit):
    """Simplified rejection sampling encoding."""
    Q = np.zeros_like(base_probs)
    for i in range(len(base_probs)):
        if i % 2 == secret_bit:
            Q[i] = base_probs[i]
    if np.sum(Q) == 0:
        Q = base_probs.copy()
    else:
        Q /= np.sum(Q)
    return Q

def steganographic_decode(token_id):
    """Decode secret bit from token id."""
    return token_id % 2

# --- FACTORY EXAMPLES (For backward compatibility with tests) ---

def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def create_predator_prey():
    sys = System("Predator-Prey Ecosystem")
# --- FACTORIES ---
def create_predator_prey():
    sys = System("Predator-Prey Ecosystem")

    def create_coelho(pos, energia=10.0):
        n = Node("Coelho", energia=energia, posição=pos, idade=0.0)
        def dynamics(self):
            self.energia -= 0.1
            self.idade += 0.01
        n.add_dynamic(dynamics)
        return n

    def create_raposa(pos, energia=15.0):
        n = Node("Raposa", energia=energia, posição=pos, idade=0.0)
        def dynamics(self):
            self.energia -= 0.2
            self.idade += 0.01
        n.add_dynamic(dynamics)
        return n

    def create_grama(pos, biomassa=100.0):
        n = Node("Grama", biomassa=biomassa, posição=pos)
        def dynamics(self):
            self.biomassa += 0.05 * (100.0 - self.biomassa)
        n.add_dynamic(dynamics)
        return n
    for _ in range(10): sys.add_node(create_coelho(np.random.rand(2) * 4))
    for _ in range(4): sys.add_node(create_raposa(np.random.rand(2) * 4))
    sys.add_node(create_grama(np.array([2.0, 2.0])))

    # Initial Population
    for _ in range(10):
        sys.add_node(create_coelho(np.random.rand(2) * 4))
    for _ in range(4):
        sys.add_node(create_raposa(np.random.rand(2) * 4))
    sys.add_node(create_grama(np.array([2.0, 2.0])))

    # Handovers
    comer_grama = Handover("ComerGrama", "Coelho", "Grama")
    comer_grama.set_condition(lambda c, g: distance(c.posição, g.posição) < 1.0)
    def comer_grama_effect(c, g):
        c.energia += 0.2 * g.biomassa
        g.biomassa -= 0.2 * g.biomassa
    comer_grama.set_effects(comer_grama_effect)
    sys.add_handover(comer_grama)

    comer_coelho = Handover("ComerCoelho", "Raposa", "Coelho")
    comer_coelho.set_condition(lambda r, c: distance(r.posição, c.posição) < 1.0)
    def comer_coelho_effect(r, c):
        r.energia += c.energia
        sys.remove_node(c)
    comer_coelho.set_effects(comer_coelho_effect)
    sys.add_handover(comer_coelho)

    return sys

def create_alcubierre_model():
    sys = System("Alcubierre Warp Drive")
    sys.metadata['R_bolha'] = 2.0
    for i in range(5):
        for j in range(5):
            r = Node("RegiãoEspaçoTempo", g=np.eye(4), x=np.array([float(i), float(j), 0.0, 0.0]), T=np.zeros((4,4)))
            sys.add_node(r)
    bubble = Node("BolhaWarp", posição=np.array([0.0, 0.0, 0.0, 0.0]), velocidade=1.5, forma=lambda r: 1.0 if r < 1.0 else 0.0)
    sys.add_node(bubble)
    def move_bubble(self): self.posição[0] += self.velocidade * 0.1
    bubble.add_dynamic(move_bubble)
    interaction = Handover("BolhaParaRegião", "BolhaWarp", "RegiãoEspaçoTempo")
    def interaction_cond(b, r): return distance(b.posição[:2], r.x[:2]) <= sys.metadata['R_bolha']
    def interaction_effect(b, r):
        dist = distance(b.posição[:2], r.x[:2])
        f_r = b.forma(dist)
        r.g[0,0] = 1.0 - (b.velocidade**2 * f_r**2)
    interaction.set_condition(interaction_cond)
    interaction.set_effects(interaction_effect)
    sys.add_handover(interaction)
    return sys

if __name__ == "__main__":
    # Test simple run
    eco = create_predator_prey()
    print(eco)
    for _ in range(10):
        eco.step()
        # Filter dead animals
        for n in eco.nodes[:]:
            if n.node_type in ["Coelho", "Raposa"] and n.energia <= 0:
                eco.remove_node(n)
        print(f"t={eco.time}: {len(eco.nodes)} nodes")
