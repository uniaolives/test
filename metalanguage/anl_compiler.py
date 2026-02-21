"""
ARKHE(N) LANGUAGE (ANL) – Python Prototype Backend
Version 0.2 - Operational Prototype with Tensor Support
"""

import numpy as np
import uuid
from typing import List, Callable, Any, Dict, Union

# --- ANL TYPE DEFINITIONS ---
# In Python, we use NumPy for vectors and tensors
Vector = np.ndarray
Tensor = np.ndarray

class Node:
    def __init__(self, node_type: str, **attributes):
        self.id = str(uuid.uuid4())[:8]
        self.node_type = node_type
        self.attributes = attributes
        self.internal_dynamics = []

    def __getattr__(self, name):
        if name in self.attributes:
            return self.attributes[name]
        raise AttributeError(f"Node {self.node_type} has no attribute {name}")

    def __setattr__(self, name, value):
        if name in ['id', 'node_type', 'attributes', 'internal_dynamics']:
            super().__setattr__(name, value)
        else:
            self.attributes[name] = value

    def add_dynamic(self, func: Callable[['Node'], None]):
        self.internal_dynamics.append(func)

    def step(self):
        for dyn in self.internal_dynamics:
            dyn(self)

    def __repr__(self):
        return f"<{self.node_type} {self.id} {self.attributes}>"

class Handover:
    def __init__(self, name: str, origin_types: Union[str, List[str]], target_types: Union[str, List[str]], protocol: str = 'CONSERVATIVE'):
        self.name = name
        self.origin_types = [origin_types] if isinstance(origin_types, str) else origin_types
        self.target_types = [target_types] if isinstance(target_types, str) else target_types
        self.protocol = protocol
        self.condition = lambda *args: True
        self.effects = lambda *args: None

    def set_condition(self, func: Callable):
        self.condition = func

    def set_effects(self, func: Callable):
        self.effects = func

    def execute(self, *nodes: Node) -> bool:
        # Simple pair-wise execution for now, but extensible to multi-node
        if len(nodes) < 2: return False

        origin = nodes[0]
        target = nodes[1]

        if origin.node_type in self.origin_types and target.node_type in self.target_types:
            if self.condition(*nodes):
                self.effects(*nodes)
                return True
        return False

class System:
    def __init__(self, name="ANL System"):
        self.name = name
        self.nodes: List[Node] = []
        self.handovers: List[Handover] = []
        self.constraints: List[Callable[['System'], bool]] = []
        self.time = 0
        self.metadata = {}

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def add_handover(self, handover: Handover):
        self.handovers.append(handover)

    def add_constraint(self, check_func: Callable[['System'], bool]):
        self.constraints.append(check_func)

    def step(self):
        # 1. Internal Dynamics
        for node in self.nodes:
            node.step()

        # 2. Handovers
        # We iterate over all pairs for pair-wise handovers
        nodes_to_check = self.nodes[:]
        for h in self.handovers:
            for i in range(len(nodes_to_check)):
                for j in range(len(nodes_to_check)):
                    if i == j: continue
                    origin = nodes_to_check[i]
                    target = nodes_to_check[j]
                    if origin in self.nodes and target in self.nodes:
                        h.execute(origin, target)

        # 3. Constraints
        for c in self.constraints:
            if not c(self):
                print(f"⚠️ Constraint violation at t={self.time}")

        self.time += 1

    def remove_node(self, node: Node):
        if node in self.nodes:
            self.nodes.remove(node)

    def __repr__(self):
        return f"System({self.name}, t={self.time}, nodes={len(self.nodes)})"

# --- UNIVERSAL ANL FUNCTIONS (Standard Library) ---

def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def integrate(value, velocity, dt):
    return value + velocity * dt

def nabla(field, pos):
    # Numerical gradient approximation
    eps = 1e-5
    grad = []
    for i in range(len(pos)):
        p_plus = pos.copy()
        p_plus[i] += eps
        p_minus = pos.copy()
        p_minus[i] -= eps
        grad.append((field(p_plus) - field(p_minus)) / (2 * eps))
    return np.array(grad)

def sample(logits, temperature=1.0):
    if temperature == 0:
        return np.argmax(logits)
    exp_logits = np.exp(logits / temperature)
    probs = exp_logits / np.sum(exp_logits)
    return np.random.choice(len(logits), p=probs)

# --- FACTORY EXAMPLES ---

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

    # Grid of spacetime regions
    for i in range(5):
        for j in range(5):
            r = Node("RegiãoEspaçoTempo",
                     g=np.eye(4),
                     x=np.array([float(i), float(j), 0.0, 0.0]),
                     T=np.zeros((4,4)))
            sys.add_node(r)

    # The Warp Bubble
    bubble = Node("BolhaWarp",
                  posição=np.array([0.0, 0.0, 0.0, 0.0]),
                  velocidade=1.5,
                  forma=lambda r: 1.0 if r < 1.0 else 0.0)
    sys.add_node(bubble)

    # Dynamics for Bubble movement
    def move_bubble(self):
        self.posição[0] += self.velocidade * 0.1 # Simplistic move along x-axis
    bubble.add_dynamic(move_bubble)

    # Handover: Bubble to Spacetime interaction
    interaction = Handover("BolhaParaRegião", "BolhaWarp", "RegiãoEspaçoTempo")
    def interaction_cond(b, r):
        return distance(b.posição[:2], r.x[:2]) <= sys.metadata['R_bolha']

    def interaction_effect(b, r):
        # Modification of metric (simulated)
        dist = distance(b.posição[:2], r.x[:2])
        f_r = b.forma(dist)
        r.g[0,0] = 1.0 - (b.velocidade**2 * f_r**2) # Metric shift

    interaction.set_condition(interaction_cond)
    interaction.set_effects(interaction_effect)
    sys.add_handover(interaction)

    return sys

if __name__ == "__main__":
    # Test simple run
    eco = create_predator_prey()
    print(eco)
    for _ in range(5):
        eco.step()
        print(f"t={eco.time}: {len(eco.nodes)} nodes")

    print("\nTesting Alcubierre model creation...")
    warp = create_alcubierre_model()
    print(warp)
    warp.step()
    print("Spacetime metric at [0,0] after 1 step:")
    print(warp.nodes[0].g)
