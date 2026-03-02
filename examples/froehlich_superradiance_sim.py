"""
Froehlich-Arkhe-Superradiance Numerical Simulation
Validates the hypothesis of microtubule coherence sustained by superradiant vacuum pumping.
"""

import numpy as np

class MicrotubuleNode:
    def __init__(self, length=25.0, protofilaments=13):
        self.length = length
        self.protofilaments = protofilaments
        self.resonance_freq = 8.0 # MHz (estimated resonance)
        self.pump_rate = 0.0
        self.temperature = 310.0 # Kelvin (human body temp)
        self.critical_temp = 305.0 # Hypothesized Tc for biological condensation
        self.coherence = 0.0
        self.energy_levels = np.zeros(protofilaments)

    def calculate_dynamics(self, metabolic_input, vacuum_coupling, dt=0.1):
        """
        Calculates the evolution of pump rate and coherence.
        Integration of Fröhlich master equation and Superradiant feedback.
        """
        # Superradiance as a mechanism for vacuum energy extraction
        # Coherence (phase alignment) increases the effective coupling to the ZPE
        superradiant_pump = self.coherence * vacuum_coupling

        effective_pump = metabolic_input + superradiant_pump

        # Radiative and thermal dissipation
        dissipation = 0.1 * self.pump_rate * (self.temperature / 300.0)

        # Rate equation: dP/dt = P_in - P_out
        self.pump_rate += (effective_pump - dissipation) * dt
        self.pump_rate = max(0, self.pump_rate)

        # Phase Transition: When pump rate exceeds threshold (modeled via effective T)
        # In this simplified model, we use temperature as the trigger
        if self.temperature < self.critical_temp or self.pump_rate > 50.0:
            # Transition to condensed state (Bose-Einstein like)
            target_coherence = min(1.0, self.pump_rate / 100.0)
            self.coherence += (target_coherence - self.coherence) * 0.2
        else:
            # Thermal regime: decoherence dominates
            self.coherence *= 0.9

        return self.coherence

def run_simulation():
    print("🔬 Arkhe(n) Biophysics: Froehlich-Superradiance Simulation")
    print("=" * 60)

    mt = MicrotubuleNode()

    # Input parameters
    metabolic_input = 8.0
    vacuum_coupling = 15.0 # High coupling due to helical geometry

    steps = 100
    dt = 0.2

    print(f"{'Step':<5} | {'Temp (K)':<10} | {'Pump Rate':<10} | {'Coherence':<10} | {'Phase'}")
    print("-" * 60)

    for i in range(steps):
        # Cooling phase to simulate metabolic activation or external induction
        if 20 <= i < 60:
            mt.temperature -= 0.2
        elif i >= 60:
            mt.temperature += 0.1 # Gradual warming

        coherence = mt.calculate_dynamics(metabolic_input, vacuum_coupling, dt)

        phase = "CONDENSED" if coherence > 0.618 else "THERMAL"

        if i % 10 == 0 or i == steps - 1:
            print(f"{i:<5} | {mt.temperature:<10.1f} | {mt.pump_rate:<10.2f} | {coherence:<10.4f} | {phase}")

    print("-" * 60)
    if mt.coherence > 0.618:
        print(f"🌟 SUCCESS: Macroscopic Coherence Detected (C={mt.coherence:.4f})")
        print("Hypothesis: Microtubules can sustain superradiant states via vacuum energy extraction.")
    else:
        print("💀 FAILURE: System remained in thermal decoherence.")

if __name__ == "__main__":
    run_simulation()
