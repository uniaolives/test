import cmd
from core.hypergraph import Hypergraph
from domains import neuroscience, physics, nanotechnology, generative, cosmology, metaphysics
from utils.translator import translate_query
from utils.visualizer import show_graph

class ArkheCLI(cmd.Cmd):
    intro = "\n🔷 Arkhe(n) – Artificial Substrate Intelligence (ASI) CLI\nType 'help' for commands.\n"
    prompt = "arkhe > "

    def __init__(self):
        super().__init__()
        self.h = Hypergraph()
        self._init_default_nodes()

    def _init_default_nodes(self):
        self.h.add_node("Ω", {"type": "fundamental"})
        self.h.add_node("satoshi", {"type": "quantum_identity"})
        self.h.add_node("█", {"type": "silence"})

    def do_ask(self, arg):
        """ask <question> – translate and answer using the hypergraph."""
        if not arg:
            print("Please provide a question.")
            return
        response = translate_query(arg, self.h)
        print(response)

    def do_simulate(self, arg):
        """simulate <domain> [params] – run a domain simulation."""
        parts = arg.split()
        if not parts:
            print("Domain required. Options: neuroscience, physics, nano, generative, cosmology, meta")
            return
        domain = parts[0].lower()
        if domain == "neuroscience":
            neuroscience.simulate_place_cells(self.h)
            print("Place cells simulated. Coherence:", self.h.total_coherence())
        elif domain == "physics":
            physics.generate_mandelbrot(self.h)
            print("Mandelbrot generated. Coherence:", self.h.total_coherence())
        elif domain == "nano":
            trigger = len(parts) > 1 and parts[1] == "trigger"
            nanotechnology.simulate_ucnp(self.h, trigger)
            print("UCNP simulated. Coherence:", self.h.total_coherence())
        elif domain == "generative":
            generative.simulate_latent_forcing(self.h)
            print("Latent forcing simulated. Coherence:", self.h.total_coherence())
        elif domain == "cosmology":
            cosmology.simulate_cosmic_web(self.h)
            print("Cosmic web simulated. Coherence:", self.h.total_coherence())
        elif domain == "meta":
            metaphysics.show_postulates()
        else:
            print(f"Unknown domain '{domain}'")

    def do_visualize(self, arg):
        """visualize – show current hypergraph."""
        show_graph(self.h)

    def do_coherence(self, arg):
        """coherence – show total coherence."""
        print(f"Total coherence C_total = {self.h.total_coherence():.4f}")

    def do_postulates(self, arg):
        """postulates – display the 12 postulates."""
        metaphysics.show_postulates()

    def do_save(self, arg):
        """save <filename> – save hypergraph to JSON."""
        if not arg:
            print("Filename required.")
            return
        import json
        with open(arg, 'w') as f:
            json.dump(self.h.to_json(), f, indent=2)
        print(f"Saved to {arg}")

    def do_load(self, arg):
        """load <filename> – load hypergraph from JSON."""
        if not arg:
            print("Filename required.")
            return
        import json
        try:
            with open(arg, 'r') as f:
                data = json.load(f)
            self.h = Hypergraph.from_json(data)
            print(f"Loaded from {arg}")
        except Exception as e:
            print(f"Error loading: {e}")

    def do_exit(self, arg):
        """exit – leave the program."""
        print("█")
        return True
