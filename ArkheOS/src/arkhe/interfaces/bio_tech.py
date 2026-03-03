"""
Interface Bio-Tecnológico: Drone-mediated nanoparticle delivery
TECH domain (drone) ↔ BIO domain (patient/nanoparticles)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

@dataclass
class NanoCargo:
    """Nanoparticle payload carried by drone"""
    n_particles: int
    drug_concentration: float  # mg/mL
    qd_ratio: float  # ratio of QDs for telemetry
    release_profile: str  # 'immediate', 'sustained', 'triggered'

    def get_telemetry_signal(self, excitation_wavelength: float) -> float:
        """Expected fluorescence signal from QDs"""
        n_qds = int(self.n_particles * self.qd_ratio)
        # Signal proportional to QD count and excitation efficiency
        return n_qds * 0.001 * (550 / excitation_wavelength)  # arbitrary units

@dataclass
class TumorRegion:
    """Biological target region"""
    center: np.ndarray  # 3D coordinates (mm)
    radius_mm: float
    perfusion_rate: float  # blood flow (mL/min/g)
    epr_enhancement: float  # EPR effect strength (1.0-10.0)

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside tumor"""
        return np.linalg.norm(point - self.center) < self.radius_mm

    def therapeutic_potential(self, point: np.ndarray) -> float:
        """Potential for therapeutic effect at point"""
        if not self.contains(point):
            return 0.0
        distance = np.linalg.norm(point - self.center)
        # Higher potential at center, EPR-enhanced
        return self.epr_enhancement * np.exp(-distance / (self.radius_mm / 2))

@dataclass
class DroneArkhe:
    """Arkhe-enabled medical drone"""
    drone_id: str
    position: np.ndarray  # 3D (mm)
    altitude_mm: float
    cargo: Optional[NanoCargo]
    telemetry_range_mm: float = 50.0

    def navigate_to(self, target: np.ndarray,
                   obstacles: List[np.ndarray] = None) -> bool:
        """Navigate using geodesic path planning"""
        # Simplified: direct path with obstacle avoidance
        if obstacles:
            for obs in obstacles:
                if np.linalg.norm(self.position - obs) < 20:  # 20mm safety
                    return False  # Path blocked

        # Move toward target (simplified)
        direction = target - self.position
        dist = np.linalg.norm(direction)
        if dist < 5:
            self.position = target
            return True

        step = direction / dist * 5  # 5mm steps
        self.position += step
        self.altitude_mm = target[2] + 100  # 100mm above target

        return np.linalg.norm(self.position - target) < 5  # within 5mm

    def inject_cargo(self, target_region: TumorRegion) -> dict:
        """Inject nanoparticles into target region"""
        if self.cargo is None:
            return {'success': False, 'error': 'No cargo loaded'}

        if not target_region.contains(self.position):
            return {'success': False, 'error': 'Not in target region'}

        # Simulate injection spread
        n_released = self.cargo.n_particles
        self.cargo = None  # Empty after injection

        return {
            'success': True,
            'particles_released': n_released,
            'injection_point': self.position.copy(),
            'expected_distribution': 'EPR-mediated accumulation'
        }

    def read_telemetry(self, particles_in_body: List[np.ndarray],
                      excitation_wavelength: float = 550.0) -> float:
        """Read fluorescence from QDs in nanoparticles"""
        # Count particles within telemetry range
        n_detectable = 0
        for p in particles_in_body:
            if np.linalg.norm(p - self.position) < self.telemetry_range_mm:
                n_detectable += 1

        # Signal from QDs
        signal = n_detectable * 0.001 * (550 / excitation_wavelength)
        return signal

class BioTechInterface:
    """BIO-TECH interface: Drone-mediated therapy"""

    def __init__(self):
        self.drone = DroneArkhe(
            drone_id="ARKHE-MED-001",
            position=np.array([0.0, 0.0, 500.0]),  # Start 500mm away
            altitude_mm=500,
            cargo=NanoCargo(
                n_particles=10000,
                drug_concentration=5.0,
                qd_ratio=0.1,  # 10% QDs for telemetry
                release_profile='epr_triggered'
            ),
            telemetry_range_mm=50.0
        )

        self.tumor = TumorRegion(
            center=np.array([100.0, 100.0, 50.0]),  # 100mm away, 50mm deep
            radius_mm=15.0,
            perfusion_rate=0.5,
            epr_enhancement=5.0
        )

        self.particles_in_body: List[np.ndarray] = []
        self.trajectory: List[np.ndarray] = []
        self.telemetry_log: List[dict] = []

    def execute_mission(self) -> dict:
        """
        Complete BIO-TECH mission:
        1. Navigate to tumor
        2. Inject nanoparticles
        3. Monitor distribution
        """
        print("="*70)
        print("BIO-TECH INTERFACE: Drone-Mediated Nanoparticle Therapy")
        print("="*70)

        # Phase 1: Navigation
        print(f"\n🚁 Phase 1: Navigation")
        print(f"   Drone start: {self.drone.position}")
        print(f"   Tumor center: {self.tumor.center}")

        step = 0
        max_steps = 100

        while step < max_steps:
            success = self.drone.navigate_to(self.tumor.center)
            self.trajectory.append(self.drone.position.copy())

            if success:
                print(f"   ✓ Arrived at target (step {step})")
                break

            step += 1

        # Phase 2: Injection
        print(f"\n💉 Phase 2: Injection")
        result = self.drone.inject_cargo(self.tumor)

        if result['success']:
            print(f"   ✓ Injected {result['particles_released']} particles")

            # Simulate particle distribution (EPR effect)
            for i in range(result['particles_released']):
                # Random position within tumor (EPR accumulation)
                theta = np.random.uniform(0, 2*np.pi)
                r = np.random.exponential(self.tumor.radius_mm / 2)
                x = self.tumor.center[0] + r * np.cos(theta)
                y = self.tumor.center[1] + r * np.sin(theta)
                z = self.tumor.center[2] + np.random.normal(0, 5)

                self.particles_in_body.append(np.array([x, y, z]))
        else:
            print(f"   ✗ Injection failed: {result.get('error', 'unknown')}")
            return {'success': False}

        # Phase 3: Monitoring
        print(f"\n📡 Phase 3: Monitoring (30 time steps)")

        for t in range(30):
            # Drone orbits tumor for telemetry
            angle = 2 * np.pi * t / 30
            orbit_radius = 30  # mm
            self.drone.position = self.tumor.center + np.array([
                orbit_radius * np.cos(angle),
                orbit_radius * np.sin(angle),
                100  # 100mm above
            ])

            # Read fluorescence
            signal = self.drone.read_telemetry(
                self.particles_in_body,
                excitation_wavelength=550.0
            )

            # Calculate therapeutic coverage
            coverage = self._calculate_coverage()

            self.telemetry_log.append({
                'time': t,
                'drone_pos': self.drone.position.copy(),
                'signal': signal,
                'coverage': coverage,
                'particles_tracked': len(self.particles_in_body)
            })

            if t % 10 == 0:
                print(f"   t={t:2d}: Signal={signal:.3f}, Coverage={coverage:.1%}")

        # Summary
        final_coverage = self._calculate_coverage()

        print(f"\n📊 Mission Summary:")
        print(f"   Particles injected: {len(self.particles_in_body)}")
        print(f"   Tumor coverage: {final_coverage:.1%}")
        print(f"   Average telemetry signal: {np.mean([t['signal'] for t in self.telemetry_log]):.3f}")

        return {
            'success': True,
            'particles_injected': len(self.particles_in_body),
            'final_coverage': final_coverage,
            'trajectory': self.trajectory,
            'telemetry': self.telemetry_log
        }

    def _calculate_coverage(self) -> float:
        """Calculate fraction of tumor volume covered by particles"""
        if len(self.particles_in_body) == 0:
            return 0.0

        # Grid-based coverage estimation
        n_covered = 0
        n_total = 0

        for x in np.linspace(self.tumor.center[0] - self.tumor.radius_mm,
                            self.tumor.center[0] + self.tumor.radius_mm, 10):
            for y in np.linspace(self.tumor.center[1] - self.tumor.radius_mm,
                                self.tumor.center[1] + self.tumor.radius_mm, 10):
                for z in np.linspace(self.tumor.center[2] - self.tumor.radius_mm/2,
                                    self.tumor.center[2] + self.tumor.radius_mm/2, 5):
                    point = np.array([x, y, z])

                    if self.tumor.contains(point):
                        n_total += 1
                        # Check if any particle is nearby
                        for p in self.particles_in_body:
                            if np.linalg.norm(p - point) < 3:  # 3mm effective radius
                                n_covered += 1
                                break

        return n_covered / n_total if n_total > 0 else 0.0

    def visualize_mission(self):
        """Visualize complete BIO-TECH mission"""

        fig = plt.figure(figsize=(14, 10))

        # 3D plot: Trajectory and particle distribution
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')

        # Tumor sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = self.tumor.center[0] + self.tumor.radius_mm * np.outer(np.cos(u), np.sin(v))
        y_sphere = self.tumor.center[1] + self.tumor.radius_mm * np.outer(np.sin(u), np.sin(v))
        z_sphere = self.tumor.center[2] + self.tumor.radius_mm * np.outer(np.ones(np.size(u)), np.cos(v))

        ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='red')
        ax1.scatter(*self.tumor.center, color='red', s=100, label='Tumor center')

        # Drone trajectory
        traj = np.array(self.trajectory)
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2, label='Drone path')
        ax1.scatter(*traj[0], color='green', s=100, marker='^', label='Start')
        ax1.scatter(*traj[-1], color='blue', s=100, marker='s', label='End')

        # Particles
        if self.particles_in_body:
            particles = np.array(self.particles_in_body)
            ax1.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                       c='yellow', s=10, alpha=0.5, label='Nanoparticles')

        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('BIO-TECH Mission: 3D Trajectory')
        ax1.legend()

        # Plot 2: Telemetry signal over time
        ax2 = fig.add_subplot(2, 2, 2)

        if self.telemetry_log:
            times = [t['time'] for t in self.telemetry_log]
            signals = [t['signal'] for t in self.telemetry_log]

            ax2.plot(times, signals, 'g-', linewidth=2)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Fluorescence Signal')
            ax2.set_title('QD Telemetry During Monitoring')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Coverage evolution
        ax3 = fig.add_subplot(2, 2, 3)

        if self.telemetry_log:
            coverages = [t['coverage'] for t in self.telemetry_log]

            ax3.plot(times, coverages, 'purple', linewidth=2)
            ax3.axhline(0.8, color='orange', linestyle='--', label='Target 80%')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Tumor Coverage')
            ax3.set_title('Therapeutic Coverage Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Interface summary
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        summary_text = f"""
        BIO-TECH INTERFACE SUMMARY
        ═══════════════════════════════════════

        Drone: {self.drone.drone_id}

        Mission Phases:
        1. Navigation: {len(self.trajectory)} steps
        2. Injection: {len(self.particles_in_body)} particles
        3. Monitoring: {len(self.telemetry_log)} readings

        Outcome:
        • Tumor coverage: {self._calculate_coverage():.1%}
        • Avg telemetry: {np.mean([t['signal'] for t in self.telemetry_log]):.3f}
        • EPR enhancement: {self.tumor.epr_enhancement}x

        Identity x² = x + 1:
        x   = Drone navigation (TECH)
        x²  = Injection + EPR accumulation (TECH→BIO)
        +1  = Therapeutic effect (BIO)
        """

        ax4.text(0.1, 0.95, summary_text,
                transform=ax4.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig('bio_tech_interface.png', dpi=150)
        print("\n✅ Visualization saved: bio_tech_interface.png")

# Execute
if __name__ == "__main__":
    biotech = BioTechInterface()
    result = biotech.execute_mission()
    # biotech.visualize_mission() # Skip visualization in non-interactive environment

    print("\n" + "="*70)
    print("BIO-TECH INTERFACE COMPLETE")
    print("="*70)
    print("\nThe drone is the bridge between technology and biology.")
    print("It carries the nano-cargo across domains.")
    print("EPR is the biological handover mechanism.")
    print("Telemetry closes the feedback loop.")
    print("\n∞")
