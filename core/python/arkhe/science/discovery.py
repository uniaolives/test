class ScienceDiscoveryEngine:
    """Orquestra a detecção de invariantes e propostas constitucionais."""
    def __init__(self, detector, constitution):
        self.detector = detector
        self.constitution = constitution

    def process_result(self, result):
        law = self.detector.observe(result)
        if law:
            self.propose_amendment(law)

    def propose_amendment(self, law):
        print(f"🔬 PROPOSING CONSTITUTIONAL AMENDMENT: {law}")
        # No sistema real, isso dispararia um processo de votação consensus
        self.constitution.add_law(law)
