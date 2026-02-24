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
        return self.identity_proof.startswith("zk-SNARK")

    def verify_signature(self) -> bool:
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
