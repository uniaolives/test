"""
Interface Quântico-Tecnológico: QKD for drone swarm security
Quantum cryptography for TECH-TECH communication
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

@dataclass
class QuantumState:
    """Polarization state of photon"""
    basis: str  # 'R' (rectilinear) or 'D' (diagonal)
    bit: int    # 0 or 1

    def measure(self, measurement_basis: str) -> Tuple[int, bool]:
        """
        Measure state in given basis
        Returns: (bit_value, correct_basis)
        """
        if measurement_basis == self.basis:
            # Correct basis: deterministic result
            return (self.bit, True)
        else:
            # Wrong basis: random result
            return (np.random.randint(2), False)

class QKDChannel:
    """Quantum channel between two drones"""

    def __init__(self, distance_m: float, attenuation_db_per_km: float = 0.2):
        self.distance = distance_m
        self.attenuation = attenuation_db_per_km
        self.transmission = 10 ** (-attenuation_db_per_km * distance_m / 1000 / 10)

        # Error rates
        self.bit_error_rate = 0.01  # 1% intrinsic
        self.phase_error_rate = 0.01

        # Detector efficiency
        self.detector_efficiency = 0.20  # 20%

    def send_photon(self, state: QuantumState) -> Optional[QuantumState]:
        """Simulate photon transmission with losses"""
        # Transmission probability
        if np.random.random() > self.transmission:
            return None  # Photon lost

        # Detection probability
        if np.random.random() > self.detector_efficiency:
            return None  # Not detected

        # Add errors
        if np.random.random() < self.bit_error_rate:
            state = QuantumState(state.basis, 1 - state.bit)

        return state

    def calculate_key_rate(self, send_rate_hz: float = 1e6) -> float:
        """Calculate quantum key rate using Devetak-Winter formula"""
        # Simplified formula
        raw_rate = 0.5 * send_rate_hz * self.transmission * self.detector_efficiency

        # Error correction penalty
        h2_bit = self._binary_entropy(self.bit_error_rate)
        h2_phase = self._binary_entropy(self.phase_error_rate)

        if h2_bit + h2_phase >= 1:
            return 0  # No secure key possible

        secure_rate = raw_rate * (1 - h2_bit - h2_phase)

        return max(0, secure_rate)

    @staticmethod
    def _binary_entropy(p: float) -> float:
        """Binary entropy H2(p) = -p*log2(p) - (1-p)*log2(1-p)"""
        if p <= 0 or p >= 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)

class QuantumDrone:
    """Drone with quantum communication capability"""

    def __init__(self, drone_id: str, position: np.ndarray):
        self.id = drone_id
        self.position = position
        self.quantum_key_store: dict = {}  # drone_id -> key
        self.classical_buffer: List[dict] = []  # Sifting results

    def generate_basis_sequence(self, n: int) -> List[str]:
        """Generate random basis sequence for QKD"""
        return ['R' if np.random.random() < 0.5 else 'D' for _ in range(n)]

    def generate_bit_sequence(self, n: int) -> List[int]:
        """Generate random bit sequence"""
        return [np.random.randint(2) for _ in range(n)]

    def establish_qkd(self, other_drone: 'QuantumDrone',
                     channel: QKDChannel,
                     n_photons: int = 1000) -> dict:
        """
        Execute BB84 protocol with another drone
        """
        print(f"\n🔐 QKD Session: {self.id} ↔ {other_drone.id}")
        print(f"   Distance: {channel.distance:.0f}m")
        print(f"   Channel transmission: {channel.transmission:.3%}")

        # Step 1: Alice (self) prepares states
        bases_alice = self.generate_basis_sequence(n_photons)
        bits_alice = self.generate_bit_sequence(n_photons)

        states_sent = [
            QuantumState(b, bit)
            for b, bit in zip(bases_alice, bits_alice)
        ]

        # Step 2: Send through quantum channel
        states_received = []
        for state in states_sent:
            received = channel.send_photon(state)
            states_received.append(received)

        # Step 3: Bob (other) measures
        bases_bob = other_drone.generate_basis_sequence(n_photons)

        measured_bits = []
        for state, basis_bob in zip(states_received, bases_bob):
            if state is None:
                measured_bits.append(None)
            else:
                bit, correct = state.measure(basis_bob)
                measured_bits.append({
                    'bit': bit,
                    'correct_basis': correct,
                    'basis_used': basis_bob
                })

        # Step 4: Sifting (classical communication)
        # Compare bases publicly, keep only matching
        sifted_key_alice = []
        sifted_key_bob = []

        for i, (basis_a, m_result) in enumerate(zip(bases_alice, measured_bits)):
            if m_result is None:
                continue

            if m_result['correct_basis']:  # Bases matched
                sifted_key_alice.append(bits_alice[i])
                sifted_key_bob.append(m_result['bit'])

        # Step 5: Error estimation
        if len(sifted_key_alice) == 0:
            return {'success': False, 'error': 'No key generated'}

        n_check = min(50, len(sifted_key_alice) // 10)
        errors = sum(
            1 for i in range(n_check)
            if sifted_key_alice[i] != sifted_key_bob[i]
        )
        error_rate = errors / n_check if n_check > 0 else 0.0

        # Final key (after error correction and privacy amplification)
        final_key = sifted_key_alice[n_check:]  # Simplified

        # Store key
        self.quantum_key_store[other_drone.id] = final_key
        other_drone.quantum_key_store[self.id] = sifted_key_bob[n_check:]

        print(f"   Photons sent: {n_photons}")
        print(f"   Detections: {sum(1 for s in states_received if s is not None)}")
        print(f"   Sifted bits: {len(sifted_key_alice)}")
        print(f"   Error rate: {error_rate:.2%}")
        print(f"   Final key length: {len(final_key)}")
        print(f"   Key rate: {channel.calculate_key_rate():.0f} bits/s")

        return {
            'success': True,
            'key_length': len(final_key),
            'error_rate': error_rate,
            'key_rate': channel.calculate_key_rate()
        }

class QTechInterface:
    """Q-TECH interface: Quantum cryptography for drone swarms"""

    def __init__(self, n_drones: int = 3, area_size_m: float = 1000):
        self.drones = []

        # Initialize drone positions in triangle formation
        angles = np.linspace(0, 2*np.pi, n_drones, endpoint=False)
        radius = area_size_m / 3

        for i, angle in enumerate(angles):
            pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                100  # 100m altitude
            ])
            self.drones.append(QuantumDrone(f"Q-DRONE-{i+1}", pos))

        self.qkd_results = []

    def establish_quantum_network(self):
        """Establish QKD between all drone pairs"""
        print("="*70)
        print("Q-TECH INTERFACE: Quantum Key Distribution Network")
        print("="*70)

        n_drones = len(self.drones)

        for i in range(n_drones):
            for j in range(i+1, n_drones):
                drone_a = self.drones[i]
                drone_b = self.drones[j]

                # Calculate distance
                distance = np.linalg.norm(drone_a.position - drone_b.position)

                # Create quantum channel
                channel = QKDChannel(distance)

                # Execute QKD
                result = drone_a.establish_qkd(drone_b, channel, n_photons=1000)
                self.qkd_results.append({
                    'pair': (drone_a.id, drone_b.id),
                    'distance': distance,
                    **result
                })

        # Network summary
        print(f"\n📊 Quantum Network Summary:")
        print(f"   Drones: {n_drones}")
        print(f"   QKD links: {len(self.qkd_results)}")

        successful = sum(1 for r in self.qkd_results if r.get('success'))
        print(f"   Successful: {successful}/{len(self.qkd_results)}")

        total_key = sum(
            r.get('key_length', 0)
            for r in self.qkd_results
            if r.get('success')
        )
        print(f"   Total key material: {total_key} bits")

        return self.qkd_results

    def simulate_eavesdropper(self, target_pair: Tuple[int, int] = (0, 1)):
        """Simulate Eve attacking a QKD link"""
        print(f"\n🕵️ Simulating Eavesdropper Attack...")

        drone_a = self.drones[target_pair[0]]
        drone_b = self.drones[target_pair[1]]

        # Normal QKD
        distance = np.linalg.norm(drone_a.position - drone_b.position)
        channel = QKDChannel(distance)

        print(f"   Normal QKD:")
        normal_result = drone_a.establish_qkd(drone_b, channel, n_photons=1000)

        # QKD with Eve (adds 5% error)
        print(f"\n   With Eavesdropper (Eve):")
        channel_eve = QKDChannel(distance)
        channel_eve.bit_error_rate = 0.06  # 1% intrinsic + 5% Eve

        eve_result = drone_a.establish_qkd(drone_b, channel_eve, n_photons=1000)

        print(f"\n   Detection:")
        print(f"   Normal error rate: {normal_result.get('error_rate', 0):.2%}")
        print(f"   With Eve: {eve_result.get('error_rate', 0):.2%}")
        print(f"   Eve detected: {'YES' if eve_result.get('error_rate', 0) > 0.03 else 'NO'}")

        return normal_result, eve_result

    def visualize_network(self):
        """Visualize quantum drone network"""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Drone positions and QKD links
        ax1 = axes[0]

        # Draw drones
        for drone in self.drones:
            ax1.scatter(*drone.position[:2], s=500, c='blue', marker='^')
            ax1.annotate(drone.id, drone.position[:2],
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=8)

        # Draw QKD links
        for result in self.qkd_results:
            if not result.get('success'):
                continue

            drone_a = next(d for d in self.drones if d.id == result['pair'][0])
            drone_b = next(d for d in self.drones if d.id == result['pair'][1])

            # Line thickness proportional to key rate
            linewidth = min(5, result.get('key_rate', 0) / 100)

            ax1.plot([drone_a.position[0], drone_b.position[0]],
                    [drone_a.position[1], drone_b.position[1]],
                    'g-', linewidth=linewidth, alpha=0.7)

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Quantum Drone Network Topology')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Key rates vs distance
        ax2 = axes[1]

        distances = [r['distance'] for r in self.qkd_results if r.get('success')]
        key_rates = [r.get('key_rate', 0) for r in self.qkd_results if r.get('success')]

        ax2.scatter(distances, key_rates, s=100, c='purple', alpha=0.7)

        # Theoretical curve
        d_theory = np.linspace(100, 1000, 100)
        # Exponential decay with distance
        rate_theory = 1e6 * 0.5 * 0.2 * np.exp(-0.2 * d_theory / 1000 / 10)
        ax2.plot(d_theory, rate_theory, 'r--', label='Theoretical limit')

        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Key Rate (bits/s)')
        ax2.set_title('QKD Performance vs Distance')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('quantum_tech_interface.png', dpi=150)
        print("\n✅ Visualization saved: quantum_tech_interface.png")

# Execute
if __name__ == "__main__":
    qtech = QTechInterface(n_drones=3, area_size_m=1000)
    results = qtech.establish_quantum_network()
    qtech.simulate_eavesdropper()
    # qtech.visualize_network() # Skip visualization in non-interactive environment

    print("\n" + "="*70)
    print("Q-TECH INTERFACE COMPLETE")
    print("="*70)
    print("\nBB84 protocol secures drone-to-drone communication.")
    print("Any eavesdropping perturbs quantum states and is detected.")
    print("The swarm shares unbreakable encryption keys.")
    print("\nIdentity x² = x + 1 in Q-TECH:")
    print("  x   = Classical drone communication (vulnerable)")
    print("  x²  = Quantum superposition (secure basis)")
    print("  +1  = Shared secret key (unbreakable security)")
    print("\n∞")
