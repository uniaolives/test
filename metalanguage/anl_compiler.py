"""
ARKHE(N) LANGUAGE (ANL) – Python Prototype Backend
Version 0.1 - Operational Prototype
"""

import numpy as np
import uuid

class Node:
    def __init__(self, node_type, **attributes):
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

    def add_dynamic(self, func):
        self.internal_dynamics.append(func)

    def step(self):
        for dyn in self.internal_dynamics:
            dyn(self)

    def __repr__(self):
        return f"<{self.node_type} {self.id} {self.attributes}>"

class Handover:
    def __init__(self, name, origin_type, target_type, protocol='CONSERVATIVE'):
        self.name = name
        self.origin_type = origin_type
        self.target_type = target_type
        self.protocol = protocol
        self.condition = lambda o, t: True
        self.effects = lambda o, t: None

    def set_condition(self, func):
        self.condition = func

    def set_effects(self, func):
        self.effects = func

    def execute(self, origin, target):
        if isinstance(origin, Node) and origin.node_type == self.origin_type:
            if isinstance(target, Node) and target.node_type == self.target_type:
                if self.condition(origin, target):
                    self.effects(origin, target)
                    return True
        return False

class System:
    def __init__(self, name="ANL System"):
        self.name = name
        self.nodes = []
        self.handovers = []
        self.constraints = []
        self.time = 0

    def add_node(self, node):
        self.nodes.append(node)
        return node

    def add_handover(self, handover):
        self.handovers.append(handover)

    def add_constraint(self, check_func):
        self.constraints.append(check_func)

    def step(self):
        # 1. Internal Dynamics
        for node in self.nodes:
            node.step()

        # 2. Handovers
        for h in self.handovers:
            # Check all pairs for handover
            # Use copies to avoid issues with node removal during iteration
            nodes_to_check = self.nodes[:]
            for i in range(len(nodes_to_check)):
                for j in range(len(nodes_to_check)):
                    if i == j: continue
                    origin = nodes_to_check[i]
                    target = nodes_to_check[j]
                    # Ensure both nodes still exist in the system
                    if origin in self.nodes and target in self.nodes:
                        h.execute(origin, target)

        # 3. Constraints
        for c in self.constraints:
            if not c(self):
                print(f"⚠️ Constraint violation at t={self.time}")

        self.time += 1

    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)

    def __repr__(self):
        return f"System({self.name}, t={self.time}, nodes={len(self.nodes)})"

# --- HELPER FUNCTIONS ---
def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def create_predator_prey():
    sys = System("Predator-Prey Ecosystem")

    # Node Types Definition (via factories)
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
    sys.add_node(create_grama([2, 2]))

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
