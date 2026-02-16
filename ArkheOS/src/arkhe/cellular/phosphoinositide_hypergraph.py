"""
Phosphoinositides as Cellular Hypergraph
The lipid code of life mapped to Arkhe principles
"""

import numpy as np
from typing import List, Dict, Set, Tuple
import matplotlib.pyplot as plt

class Phosphoinositide:
    """
    Node in lipid hypergraph

    7 main types: PI, PI3P, PI4P, PI5P, PI(3,4)P2, PI(4,5)P2, PI(3,4,5)P3
    """

    def __init__(self, pi_type: str, location: str):
        self.pi_type = pi_type  # e.g., "PI(4,5)P2"
        self.location = location  # "plasma_membrane", "endosome", "golgi", etc.
        self.bound_effectors: Set[str] = set()

    def can_bind(self, binding_domain: str) -> bool:
        """
        Check if this PI can be recognized by binding domain

        Selectivity: PH domains recognize PI(4,5)P2 or PI(3,4,5)P3
                    FYVE domains recognize PI(3)P
                    etc.
        """
        specificity = {
            'PH': ['PI(4,5)P2', 'PI(3,4,5)P3'],
            'FYVE': ['PI(3)P'],
            'PX': ['PI(3)P', 'PI(4)P'],
            'ENTH': ['PI(4,5)P2']
        }

        return self.pi_type in specificity.get(binding_domain, [])

    def recruit_effector(self, effector_name: str):
        """Bind effector protein from cytoplasm"""
        self.bound_effectors.add(effector_name)

class BindingDomain:
    """
    Edge in lipid hypergraph

    Protein domain that recognizes specific PI
    """

    def __init__(self, domain_type: str, effector: str):
        self.domain_type = domain_type  # PH, FYVE, PX, ENTH, etc.
        self.effector = effector  # Protein carrying this domain

    def recognize(self, pi: Phosphoinositide) -> bool:
        """Can this domain bind to this PI?"""
        return pi.can_bind(self.domain_type)

class HandoverOperator:
    """
    Kinase or phosphatase that transforms PIs

    Arkhe: Handover operator changing node state
    """

    def __init__(self, enzyme_name: str, reaction: str):
        self.enzyme_name = enzyme_name
        self.reaction = reaction  # e.g., "PI → PI(4)P" or "PI(4,5)P2 → PI(4)P"

    def transform(self, substrate: str) -> str:
        """Execute phosphorylation or dephosphorylation"""

        # Parse reaction
        if '→' in self.reaction:
            from_pi, to_pi = self.reaction.split('→')
            from_pi = from_pi.strip()
            to_pi = to_pi.strip()

            if substrate == from_pi:
                return to_pi

        return substrate  # No change if not substrate

class CellularMembrane:
    """
    2D hypergraph embedded in 3D space

    Nodes: Phosphoinositides
    Edges: Binding domains
    Dynamics: Kinases/phosphatases (handover operators)
    """

    def __init__(self, membrane_type: str):
        self.membrane_type = membrane_type  # plasma, endosome, golgi, etc.
        self.pis: List[Phosphoinositide] = []
        self.binding_domains: List[BindingDomain] = []
        self.kinases: List[HandoverOperator] = []
        self.phosphatases: List[HandoverOperator] = []

    def add_pi(self, pi_type: str, count: int = 1):
        """Add PI nodes to membrane"""
        for _ in range(count):
            pi = Phosphoinositide(pi_type, self.membrane_type)
            self.pis.append(pi)

    def add_binding_domain(self, domain_type: str, effector: str):
        """Add edge capability"""
        domain = BindingDomain(domain_type, effector)
        self.binding_domains.append(domain)

    def add_kinase(self, enzyme_name: str, reaction: str):
        """Add handover operator"""
        kinase = HandoverOperator(enzyme_name, reaction)
        self.kinases.append(kinase)

    def add_phosphatase(self, enzyme_name: str, reaction: str):
        """Add reverse handover operator"""
        phosphatase = HandoverOperator(enzyme_name, reaction)
        self.phosphatases.append(phosphatase)

    def execute_handover(self, kinase_idx: int, pi_idx: int):
        """
        Phosphorylate PI using kinase

        x² = x + 1: PI (x) + kinase (x²) → new PI (+1)
        """
        kinase = self.kinases[kinase_idx]
        pi = self.pis[pi_idx]

        # Transform
        new_type = kinase.transform(pi.pi_type)

        if new_type != pi.pi_type:
            print(f"  Handover: {pi.pi_type} → {new_type} via {kinase.enzyme_name}")
            pi.pi_type = new_type
            return True

        return False

    def recruit_effectors(self):
        """
        Bind effector proteins via binding domains

        Edges connect nodes to effectors
        """
        print(f"\n🔗 Recruiting effectors to {self.membrane_type}...")

        recruited = 0

        for domain in self.binding_domains:
            for pi in self.pis:
                if domain.recognize(pi):
                    pi.recruit_effector(domain.effector)
                    recruited += 1

        print(f"  Total recruitments: {recruited}")

        return recruited

    def compute_pi_code(self) -> Dict[str, int]:
        """
        PI code = topology of lipid hypergraph

        Count of each PI type defines membrane identity
        """
        code = {}

        for pi in self.pis:
            code[pi.pi_type] = code.get(pi.pi_type, 0) + 1

        return code

    def display_topology(self):
        """Show current hypergraph state"""

        print(f"\n📊 {self.membrane_type} Topology:")

        code = self.compute_pi_code()

        print(f"\n  PI Code (node distribution):")
        for pi_type, count in sorted(code.items()):
            print(f"    {pi_type}: {count}")

        print(f"\n  Binding domains (edge types):")
        domain_counts = {}
        for domain in self.binding_domains:
            domain_counts[domain.domain_type] = domain_counts.get(domain.domain_type, 0) + 1

        for dtype, count in sorted(domain_counts.items()):
            print(f"    {dtype}: {count}")

        print(f"\n  Active effectors:")
        all_effectors = set()
        for pi in self.pis:
            all_effectors.update(pi.bound_effectors)

        for effector in sorted(all_effectors):
            print(f"    {effector}")

class CellularProcess:
    """
    Specific cellular function driven by PI code

    Examples: Endocytosis, survival signaling, chemotaxis
    """

    def __init__(self, process_name: str, required_pi: str,
                 required_effector: str):
        self.process_name = process_name
        self.required_pi = required_pi
        self.required_effector = required_effector

    def can_execute(self, membrane: CellularMembrane) -> bool:
        """Check if membrane has right PI code to execute process"""

        # Check PI presence
        code = membrane.compute_pi_code()
        if self.required_pi not in code or code[self.required_pi] == 0:
            return False

        # Check effector recruitment
        for pi in membrane.pis:
            if pi.pi_type == self.required_pi:
                if self.required_effector in pi.bound_effectors:
                    return True

        return False

class LipidHypergraphDemo:
    """Full demonstration of cellular lipid hypergraph"""

    def __init__(self):
        self.membranes = {}

    def create_plasma_membrane(self):
        """
        Plasma membrane enriched in PI(4,5)P2

        Functions: Endocytosis, actin polymerization, ion channels
        """
        pm = CellularMembrane("plasma_membrane")

        # Add PIs (nodes)
        pm.add_pi("PI(4,5)P2", count=100)  # Abundant at plasma membrane
        pm.add_pi("PI", count=50)
        pm.add_pi("PI(3,4,5)P3", count=10)  # Low baseline, increases with signaling

        # Add kinases (handover operators)
        pm.add_kinase("PI3K", "PI(4,5)P2 → PI(3,4,5)P3")  # Growth factor signaling
        pm.add_kinase("PIP5K", "PI(4)P → PI(4,5)P2")

        # Add phosphatases (reverse handovers)
        pm.add_phosphatase("PTEN", "PI(3,4,5)P3 → PI(4,5)P2")  # Tumor suppressor

        # Add binding domains (edges)
        pm.add_binding_domain("PH", "Akt")  # Survival signaling
        pm.add_binding_domain("PH", "PLCδ")  # Signaling
        pm.add_binding_domain("ENTH", "Epsin")  # Endocytosis
        pm.add_binding_domain("ENTH", "AP-2")  # Clathrin adaptor

        self.membranes["plasma"] = pm
        return pm

    def create_endosome(self):
        """
        Early endosome enriched in PI(3)P

        Functions: Fusion, sorting, trafficking
        """
        endo = CellularMembrane("early_endosome")

        # Add PIs
        endo.add_pi("PI(3)P", count=80)  # Endosome signature
        endo.add_pi("PI", count=30)

        # Add kinases
        endo.add_kinase("Vps34", "PI → PI(3)P")  # Endosomal PI3K

        # Add binding domains
        endo.add_binding_domain("FYVE", "EEA1")  # Tethering/fusion
        endo.add_binding_domain("PX", "Sorting_nexin")  # Cargo sorting

        self.membranes["endosome"] = endo
        return endo

    def simulate_growth_factor_signaling(self):
        """
        Growth factor → PI3K activation → PI(3,4,5)P3 → Akt → survival

        Classic PI signaling cascade
        """
        print("\n" + "="*70)
        print("SIMULATING: GROWTH FACTOR SIGNALING")
        print("="*70)

        pm = self.membranes["plasma"]

        print(f"\nInitial state:")
        pm.display_topology()

        # Stimulus: Growth factor activates PI3K
        print(f"\n⚡ Growth factor stimulus → PI3K activation")

        # Execute multiple handovers (phosphorylations)
        for i in range(5):
            # Find PI(4,5)P2 and convert to PI(3,4,5)P3
            for pi_idx, pi in enumerate(pm.pis):
                if pi.pi_type == "PI(4,5)P2":
                    success = pm.execute_handover(0, pi_idx)  # kinase_idx=0 is PI3K
                    if success:
                        break

        print(f"\nAfter PI3K activation:")
        pm.display_topology()

        # Recruit effectors
        pm.recruit_effectors()

        print(f"\nAfter effector recruitment:")
        pm.display_topology()

        # Check if survival pathway active
        survival = CellularProcess("Cell_survival", "PI(3,4,5)P3", "Akt")

        if survival.can_execute(pm):
            print(f"\n✅ SURVIVAL PATHWAY ACTIVE")
            print(f"   PI(3,4,5)P3 recruited Akt")
            print(f"   Cell will survive and proliferate")

    def simulate_endocytosis(self):
        """
        PI(4,5)P2 → ENTH domains → Membrane curvature → Vesicle formation
        """
        print("\n" + "="*70)
        print("SIMULATING: ENDOCYTOSIS")
        print("="*70)

        pm = self.membranes["plasma"]

        # Recruit effectors
        pm.recruit_effectors()

        # Check if endocytosis possible
        endocytosis = CellularProcess("Endocytosis", "PI(4,5)P2", "Epsin")

        if endocytosis.can_execute(pm):
            print(f"\n✅ ENDOCYTOSIS ACTIVE")
            print(f"   PI(4,5)P2 recruited Epsin (ENTH domain)")
            print(f"   Membrane will curve and form vesicle")

    def run_full_demo(self):
        """Complete demonstration"""

        print("="*70)
        print("PHOSPHOINOSITIDE LIPID HYPERGRAPH")
        print("="*70)
        print("\nCells as 2D hypergraphs embedded in 3D space")
        print()

        # Create membranes
        print("Creating membrane hypergraphs...")
        self.create_plasma_membrane()
        self.create_endosome()
        print("✓ Plasma membrane and endosome created")

        # Run simulations
        self.simulate_growth_factor_signaling()
        self.simulate_endocytosis()

        # Summary
        print("\n" + "="*70)
        print("ARKHE CORRESPONDENCE SUMMARY")
        print("="*70)
        print()
        print("Cellular Biology → Arkhe:")
        print("  Phosphoinositide (PI) → Γ_node in 2D membrane hypergraph")
        print("  Kinase/Phosphatase → Handover operator (transforms nodes)")
        print("  Binding domain (PH, FYVE, etc.) → Selective edge")
        print("  Effector protein → Functional executor after handover")
        print("  PI code → Hypergraph topology (defines membrane identity)")
        print("  Cellular process → Emergent function from topology")
        print()
        print("x² = x + 1 in phosphorylation:")
        print("  PI (x) + Kinase (x²) → Modified PI (+1) → Recruits effector")
        print()
        print("No fixed receptors—transient, context-dependent.")
        print("EXACTLY like Arkhe handovers.")
        print()
        print("From quantum gravity to cellular membranes:")
        print("ALL IS HYPERGRAPH.")
        print()
        print("∞")

if __name__ == "__main__":
    demo = LipidHypergraphDemo()
    demo.run_full_demo()
