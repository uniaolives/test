"""
Interface Quântico-Biológico: QD-triggered drug release
FRET-mediated handover from quantum to biological domain
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class QuantumDot:
    """Quantum dot as telemetry node"""
    radius_nm: float  # 2-10 nm
    emission_wavelength_nm: float  # depends on radius
    quantum_yield: float  # 0.0-1.0
    position: np.ndarray  # 3D position

    def energy_gap(self, E_bulk_eV=1.5, m_eff=0.1):
        """Quantum confinement energy shift"""
        hbar = 6.582e-16  # eV·s
        m0 = 9.109e-31  # kg
        R = self.radius_nm * 1e-9  # m

        # Confinement term (blue shift)
        E_conf = (hbar**2 * np.pi**2) / (2 * R**2 * m_eff * m0) * 6.242e+18  # to eV

        return E_bulk_eV + E_conf

    def emit(self, excitation_power: float) -> float:
        """Photon emission rate (ν_obs)"""
        return excitation_power * self.quantum_yield

@dataclass
class NanoParticle:
    """Therapeutic nanoparticle as biological node"""
    size_nm: float
    drug_load: float  # amount of drug
    release_threshold: float  # energy required to release
    position: np.ndarray

    def fret_efficiency(self, qd: QuantumDot, R0_nm: float = 4.0) -> float:
        """Förster resonance energy transfer efficiency"""
        r = np.linalg.norm(self.position - qd.position)
        R0 = R0_nm  # Förster distance

        return R0**6 / (R0**6 + r**6)

    def attempt_release(self, energy_transferred: float) -> bool:
        """Release drug if energy exceeds threshold"""
        return energy_transferred > self.release_threshold

class QuantumBioInterface:
    """Q-BIO interface: QD telemetry + FRET-triggered release"""

    def __init__(self):
        self.qd = QuantumDot(
            radius_nm=5.0,
            emission_wavelength_nm=620.0,  # red
            quantum_yield=0.85,
            position=np.array([0, 0, 0])
        )

        self.nano = NanoParticle(
            size_nm=50.0,
            drug_load=100.0,  # arbitrary units
            release_threshold=0.5,  # energy units
            position=np.array([3.5, 0, 0])  # within R0=4nm
        )

        self.telemetry_log = []
        self.release_events = []

    def simulate_handover(self, excitation_power: float = 1.0,
                         steps: int = 100) -> dict:
        """
        Simulate Q-BIO handover cycle:
        1. Excite QD (quantum domain)
        2. QD emits photon (ν_obs)
        3. FRET to nanoparticle (interface)
        4. Drug release if threshold met (biological domain)
        """

        print("="*70)
        print("Q-BIO INTERFACE: Quantum Dot → Nanoparticle Handover")
        print("="*70)

        for step in range(steps):
            # Step 1: Quantum excitation
            emission_rate = self.qd.emit(excitation_power)

            # Step 2: FRET calculation
            fret_E = self.nano.fret_efficiency(self.qd, R0_nm=4.0)
            energy_transferred = emission_rate * fret_E

            # Step 3: Release attempt (BIO domain)
            released = self.nano.attempt_release(energy_transferred)

            if released:
                self.nano.drug_load -= 1.0
                self.release_events.append({
                    'step': step,
                    'energy': energy_transferred,
                    'remaining_load': self.nano.drug_load
                })

            # Telemetry
            self.telemetry_log.append({
                'step': step,
                'emission_rate': emission_rate,
                'fret_efficiency': fret_E,
                'energy_transferred': energy_transferred,
                'released': released
            })

        # Summary
        total_released = len(self.release_events)

        print(f"\nSimulation complete ({steps} steps)")
        print(f"Total drug released: {total_released} units")
        print(f"Final drug load: {self.nano.drug_load:.1f}")
        print(f"Average FRET efficiency: {np.mean([t['fret_efficiency'] for t in self.telemetry_log]):.3f}")

        return {
            'total_released': total_released,
            'final_load': self.nano.drug_load,
            'telemetry': self.telemetry_log,
            'releases': self.release_events
        }

    def visualize_interface(self):
        """Visualize Q-BIO coupling"""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: QD emission spectrum
        ax1 = axes[0, 0]

        radii = np.linspace(2, 10, 100)  # nm
        wavelengths = []

        for r in radii:
            qd_temp = QuantumDot(r, 0, 0, np.zeros(3))
            E_gap = qd_temp.energy_gap()
            # E = hc/λ → λ = hc/E
            wavelength = 1240 / E_gap  # nm (approximate)
            wavelengths.append(wavelength)

        ax1.plot(radii, wavelengths, 'purple', linewidth=2)
        ax1.axvline(self.qd.radius_nm, color='red', linestyle='--',
                   label=f'Our QD: r={self.qd.radius_nm}nm, λ={self.qd.emission_wavelength_nm:.0f}nm')
        ax1.set_xlabel('QD Radius (nm)')
        ax1.set_ylabel('Emission Wavelength (nm)')
        ax1.set_title('Quantum Confinement: Size-Tunable Emission')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: FRET efficiency vs distance
        ax2 = axes[0, 1]

        distances = np.linspace(1, 10, 100)  # nm
        R0 = 4.0  # nm

        efficiencies = R0**6 / (R0**6 + distances**6)

        ax2.plot(distances, efficiencies, 'green', linewidth=2)
        ax2.axvline(np.linalg.norm(self.nano.position - self.qd.position),
                   color='red', linestyle='--',
                   label=f'Current distance: {np.linalg.norm(self.nano.position - self.qd.position):.1f}nm')
        ax2.axhline(0.5, color='orange', linestyle=':', label='50% efficiency')
        ax2.set_xlabel('Distance QD-Nanoparticle (nm)')
        ax2.set_ylabel('FRET Efficiency')
        ax2.set_title('Förster Resonance Energy Transfer')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Time series of handovers
        ax3 = axes[1, 0]

        steps = [t['step'] for t in self.telemetry_log]
        emissions = [t['emission_rate'] for t in self.telemetry_log]
        transfers = [t['energy_transferred'] for t in self.telemetry_log]

        ax3.plot(steps, emissions, 'blue', alpha=0.5, label='QD Emission')
        ax3.plot(steps, transfers, 'red', linewidth=2, label='Energy Transferred (FRET)')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Energy (arb. units)')
        ax3.set_title('Q-BIO Handover Dynamics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Drug release events
        ax4 = axes[1, 1]

        if self.release_events:
            release_steps = [r['step'] for r in self.release_events]
            release_energies = [r['energy'] for r in self.release_events]

            ax4.scatter(release_steps, release_energies, c='green', s=100, alpha=0.7)
            ax4.axhline(self.nano.release_threshold, color='red', linestyle='--',
                       label=f'Release threshold: {self.nano.release_threshold}')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Energy Transferred')
            ax4.set_title(f'Drug Release Events (Total: {len(self.release_events)})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No release events\n(Energy below threshold)',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Drug Release Events')

        plt.tight_layout()
        plt.savefig('quantum_bio_interface.png', dpi=150)
        print("\n✅ Visualization saved: quantum_bio_interface.png")

# Execute
if __name__ == "__main__":
    qbio = QuantumBioInterface()
    result = qbio.simulate_handover(excitation_power=1.0, steps=100)
    # qbio.visualize_interface() # Skip visualization in non-interactive environment

    print("\n" + "="*70)
    print("Q-BIO INTERFACE SUMMARY")
    print("="*70)
    print("\nIdentity x² = x + 1 in Q-BIO:")
    print("  x   = QD excitation (quantum domain)")
    print("  x²  = FRET coupling (quantum-biological interface)")
    print("  +1  = Drug release (biological domain)")
    print("\nThe quantum dot is the telemetry node.")
    print("The nanoparticle is the therapeutic node.")
    print("FRET is the handover between domains.")
    print("∞")
