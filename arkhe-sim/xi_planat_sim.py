# arkhe-sim/xi_planat_sim.py
import math

class XiPlanatNode:
    def __init__(self, p, q, l_prime=1.088):
        self.p = p
        self.q = q
        self.n = p**2 + q**2          # Gaussian Norm N
        self.l_prime = l_prime        # Arithmetic Height
        self.m0 = 1.0
        self.alpha = 0.5

        # Unification formulas
        self.mass = (self.m0 * math.sqrt(self.n)) / self.l_prime
        self.coherence = 1.0 / (1.0 + self.alpha * self.l_prime)

    def __str__(self):
        return f"Node(p={self.p}, q={self.q}) | N={self.n} | Mass={self.mass:.4f} | λ₂={self.coherence:.4f}"

def is_sum_of_two_squares(n):
    if n == 0: return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                count += 1
                temp //= d
            if d % 4 == 3 and count % 2 != 0:
                return False
        d += 1
    if temp > 1 and temp % 4 == 3:
        return False
    return True

def run_simulation():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Ω+247: Xi-Planat Unification Simulation                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    l_prime_mt = 1.088 # Microtubule curve E200b2
    print(f"Using Arithmetic Height L'(E,1) = {l_prime_mt}\n")

    print("Mode Spectrum (N = p² + q²):")
    valid_modes = []
    for n in range(1, 21):
        if is_sum_of_two_squares(n):
            # Find a pair (p,q)
            found = False
            for p in range(int(math.sqrt(n)) + 1):
                q_sq = n - p**2
                q = int(math.sqrt(q_sq))
                if q*q == q_sq:
                    node = XiPlanatNode(p, q, l_prime_mt)
                    valid_modes.append(node)
                    print(f"  [ALLOWED]  {node}")
                    found = True
                    break
        else:
            print(f"  [FORBIDDEN] N={n} (Gap)")

    print("\nMass Hierarchy Analysis:")
    for node in valid_modes[:5]:
        print(f"  m_Ξ({node.n}) / m_0 = {node.mass:.4f}")

    print("\nCoherence Condition:")
    print(f"  Global Coherence λ₂ = {valid_modes[0].coherence:.4f}")
    if valid_modes[0].coherence > 0.6:
        print("  ✅ System is in the Orquestration regime.")
    else:
        print("  ⚠️ Low coherence detected.")

if __name__ == "__main__":
    run_simulation()
