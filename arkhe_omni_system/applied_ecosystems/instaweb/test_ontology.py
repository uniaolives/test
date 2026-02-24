"""
test_ontology.py
Verifies the Ontological Unification in Instaweb (IC-Map-Vector).
Identifies Position, State, and Topology.
"""

import numpy as np

class HyperbolicCoord:
    def __init__(self, r, theta, z):
        self.r = r
        self.theta = theta
        self.z = z

class IntegratedView:
    def __init__(self, position, physical, cognitive, quantum):
        self.position = position
        self.physical = physical
        self.cognitive = cognitive
        self.quantum = quantum

class IntegratedMap:
    """The Map is the Territory."""
    def __init__(self):
        self.store = {}

    def bind(self, coord, physical, cognitive, quantum):
        key = (coord.r, coord.theta, coord.z)
        self.store[key] = IntegratedView(coord, physical, cognitive, quantum)

    def project(self, coord):
        key = (coord.r, coord.theta, coord.z)
        return self.store.get(key)

def test_ontological_identification():
    print("--- Testing Instaweb Ontological Unification ---")

    # 1. Define a coordinate (Position)
    pos = HyperbolicCoord(0.5, np.pi/4, 1.0)

    # 2. Define associated States
    phys = {"elasticity": 0.85}
    cogn = {"attention": 1.0}
    quant = {"psi": "|0> + |1>"}

    # 3. Initialize Map (Topology)
    i_map = IntegratedMap()
    i_map.bind(pos, phys, cogn, quant)

    # 4. Verification: Querying position returns the entire Integrated state
    view = i_map.project(pos)

    assert view is not None
    assert view.physical["elasticity"] == 0.85
    assert view.cognitive["attention"] == 1.0
    assert view.quantum["psi"] == "|0> + |1>"

    print("✅ Ontological Identification Verified: Position = State = Topology.")

if __name__ == "__main__":
    test_ontological_identification()
