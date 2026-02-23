"""
ARKHE(N) LANGUAGE (ANL) – Python Core Module
Version 0.7 - Active Inference, Ouroboros Loops & ArkheProtocol
"""

import numpy as np
import uuid
import json
import hashlib
from typing import List, Callable, Any, Dict, Union, Optional

# --- ANL PROTOCOLS ---
class Protocol:
    CONSERVATIVE = 'CONSERVATIVE'
    CREATIVE = 'CREATIVE'
    DESTRUCTIVE = 'DESTRUCTIVE'
    TRANSMUTATIVE = 'TRANSMUTATIVE'
    ASYNCHRONOUS = 'ASYNCHRONOUS'
    TRANSMUTATIVE_ABSOLUTE = 'TRANSMUTATIVE_ABSOLUTE'

# --- ARKHE PROTOCOL STRUCTURES ---

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
        self.capabilities = attributes.get('capabilities', [])

        # v0.7 Active Inference / Learning
        self.beliefs = np.array([0.5, 0.5]) # Categorical distribution
        self.dirichlet_params = np.ones(2)   # Dirichlet counters
        self.curiosity_score = 0.0

    def __getattr__(self, name):
        if name in self.attributes:
            return self.attributes[name]
        raise AttributeError(f"Node {self.node_type} has no attribute {name}")

    def __setattr__(self, name, value):
        if name in ['id', 'node_type', 'attributes', 'internal_dynamics', 'events', 'is_asi', 'beliefs', 'dirichlet_params', 'curiosity_score', 'capabilities']:
            super().__setattr__(name, value)
        else:
            self.attributes[name] = value

    def add_dynamic(self, func: Callable[['Node'], None]):
        self.internal_dynamics.append(func)

    def trigger_event(self, event_name: str, payload: Any = None):
        self.events.append((event_name, payload))

    def step(self):
        # Active Inference: Minimize Variational Free Energy
        self.minimize_vfe()
        for dyn in self.internal_dynamics:
            dyn(self)

    def minimize_vfe(self):
        """Mock VFE minimization: update beliefs based on observations."""
        # In a real FEP model, this would be a gradient descent on VFE
        observation_index = 0 if np.random.random() > 0.5 else 1
        self.dirichlet_params[observation_index] += 0.1
        self.beliefs = self.dirichlet_params / np.sum(self.dirichlet_params)

    def calculate_epistemic_value(self):
        """Pure Curiosity: Information gain from Dirichlet updates."""
        self.curiosity_score = np.sum(np.abs(np.log(self.dirichlet_params / np.sum(self.dirichlet_params))))
        return self.curiosity_score

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
        if len(nodes) == 2:
            if not self.target_types: return False
            origin, target = nodes
            if origin.node_type in self.origin_types and target.node_type in self.target_types:
                if self.condition(*nodes):
                    self.effects(*nodes)
                    return True
        elif len(nodes) == 1:
            if self.target_types: return False
            origin = nodes[0]
            if origin.node_type in self.origin_types:
                if self.condition(*nodes):
                    self.effects(*nodes)
                    return True
        return False

class ArkheLink(Handover):
    def __init__(self, source: Node, target: Node, intent: IntentObject, context: ContextSnapshot, ontology: str):
        super().__init__("ArkheLink", source.node_type, target.node_type, Protocol.TRANSMUTATIVE)
        self.source_node = source
        self.target_node = target
        self.identity_proof = "zk-SNARK-placeholder"
        self.intent = intent
        self.context = context
        self.ontology = ontology
        self.signature = "ed25519-placeholder"
        self.preconditions = []
        self.postconditions = []

    def verify_identity(self) -> bool:
        # Mock zk-SNARK verification
        print(f"  [Security] Verifying zk-SNARK for {self.source_node.id}...")
        return self.identity_proof.startswith("zk-SNARK")

    def verify_signature(self) -> bool:
        # Mock signature verification
        print(f"  [Security] Verifying ed25519 signature for handover {self.name}...")
        return self.signature.startswith("ed25519")

    def execute(self) -> bool:
        # Refined execution for Link Layer
        print(f"--- Handover Attempt: {self.name} ({self.source_node.node_type} -> {self.target_node.node_type}) ---")

        if not self.verify_identity():
            print("  [Failed] Identity verification failed.")
            return False
        if not self.verify_signature():
            print("  [Failed] Signature verification failed.")
            return False

        # Check preconditions
        print("  [LinkLayer] Checking preconditions...")
        for pre in self.preconditions:
            if not pre(self.source_node, self.target_node):
                print("  [Failed] Precondition not met.")
                return False

        # Execute effects
        print("  [LinkLayer] Executing effects...")
        self.effects(self.source_node, self.target_node)

        # Check postconditions
        print("  [LinkLayer] Validating postconditions and metrics...")
        for post in self.postconditions:
            if not post(self.source_node, self.target_node):
                print("  [Failed] Postcondition not met.")
                return False

        # Check metrics
        all_metrics_passed = True
        for metric in self.intent.success_metrics:
            if not metric.satisfied():
                print(f"  [Metric Failed] {metric.name}: current={metric.value}, threshold={metric.threshold}")
                all_metrics_passed = False
            else:
                print(f"  [Metric Passed] {metric.name}: {metric.value} >= {metric.threshold}")

        if all_metrics_passed:
            print("  [Success] Handover completed successfully.")
        else:
            print("  [Partial] Handover completed but some metrics failed.")

        return True

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
        self.coherence = 1.0

    def discover_agents(self, goal: str) -> List[Node]:
        """Network Layer: Discover agents based on goal matching capabilities."""
        return [n for n in self.nodes if goal in n.capabilities]

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

    def step(self):
        # 1. Internal Dynamics
        for node in self.nodes:
            node.step()

        # 2. Handovers
        for h in self.handovers:
            # Binary
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if i == j: continue
                    h.execute(self.nodes[i], self.nodes[j])
            # Unary
            for n in self.nodes:
                h.execute(n)

        # 3. Ouroboros Loop (Feedback)
        self.ouroboros_feedback()

        # 4. Global Dynamics
        for dyn in self.global_dynamics:
            dyn(self)

        # 5. Constraints
        for c in self.constraints:
            if not c["check"](self):
                if c["mode"] == ConstraintMode.INVIOLABLE_AXIOM:
                    raise RuntimeError(f"Axiom Breached in {self.name}")

        self.time += 1

    def ouroboros_feedback(self):
        """Ouroboros: System's state affects its own parameters."""
        avg_curiosity = np.mean([n.calculate_epistemic_value() for n in self.nodes]) if self.nodes else 0
        self.coherence = 0.9 * self.coherence + 0.1 * (1.0 / (1.0 + avg_curiosity))

    def remove_node(self, node: Node):
        if node in self.nodes: self.nodes.remove(node)

# --- UTILS ---
def kl_divergence(p, q):
    p = np.where(p == 0, 1e-12, p)
    q = np.where(q == 0, 1e-12, q)
    return np.sum(p * np.log(p / q))

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
