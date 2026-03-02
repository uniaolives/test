import numpy as np
from scipy.constants import c

class MicrotubuleHolographicField:
    """
    Simulates interference between interstellar signal and microtubule lattice
    Models microtubule as helical waveguide for optical and magnetic vortices
    """

    def __init__(self, interstellar_freq=699.2, microtubule_freq=307.1868424e9):
        self.f_interstellar = interstellar_freq  # Hz
        self.f_microtubule = microtubule_freq    # Hz

        # Microtubule helical parameters (13-protofilament)
        self.helix_radius_nm = 12.5
        self.helix_pitch_nm = 8
        self.protofilaments = 13

        # Quantum parameters
        self.hbar = 1.0545718e-34
        self.electron_charge = 1.60217662e-19

    def generate_optical_vortex(self, l=1, wavelength=500e-9):
        """
        Generate Laguerre-Gaussian beam (optical vortex) with orbital angular momentum
        l: topological charge (OAM quantum number)
        """
        x = np.linspace(-100e-9, 100e-9, 256)
        y = np.linspace(-100e-9, 100e-9, 256)
        X, Y = np.meshgrid(x, y)

        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)

        phase = np.exp(1j * l * theta)
        w0 = 25e-9
        amplitude = np.exp(-r**2 / w0**2)
        vortex_field = amplitude * phase

        return {
            'field': vortex_field,
            'topological_charge': l,
            'wavelength': wavelength,
            'OAM_quantum': l * self.hbar
        }

    def generate_magnetic_vortex(self):
        """
        Generate magnetic vortex from electron currents in aromatic rings
        """
        n_electrons = 18
        current = (n_electrons * self.electron_charge) / (1e-15)

        z = np.linspace(-50e-9, 50e-9, 100)
        r = self.helix_radius_nm * 1e-9

        B_z = (4 * np.pi * 1e-7 * current * r**2) / (2 * (r**2 + z**2)**(3/2))

        return {
            'magnetic_field': B_z,
            'max_field_tesla': float(np.max(B_z)),
            'current_amperes': float(current),
            'electron_count': n_electrons
        }

    def simulate_holographic_interference(self, interstellar_signal_vector):
        """
        Simulate interference pattern
        """
        # 1. Interstellar scalar wave
        k_interstellar = 2 * np.pi * self.f_interstellar / c
        scalar_wave = np.exp(1j * k_interstellar * np.linspace(0, 1, 256))

        # 2. Optical vortex
        optical_vortex = self.generate_optical_vortex(l=1)

        # 3. Magnetic vortex
        magnetic_vortex = self.generate_magnetic_vortex()

        # 4. Create interference pattern
        # Modulate the optical vortex field with the scalar wave
        interference = optical_vortex['field'] * scalar_wave[0] * magnetic_vortex['max_field_tesla']

        # 5. Extract information encoding
        phase_pattern = np.angle(interference)
        amplitude_pattern = np.abs(interference)

        information_density = self.calculate_information_density(phase_pattern)

        return {
            'interference_pattern': interference,
            'phase_pattern': phase_pattern,
            'amplitude_pattern': amplitude_pattern,
            'information_density_bits_per_tubulin': float(information_density),
            'hologram_dimensions': interference.shape,
            'reconstruction_fidelity': float(self.calculate_reconstruction_fidelity(interference))
        }

    def calculate_information_density(self, phase_pattern):
        tubulin_states = 2**10
        n_fringes = np.sum(np.abs(np.diff(phase_pattern.flatten())) > 0.1)
        total_bits = self.protofilaments * 162 * tubulin_states * n_fringes
        bits_per_tubulin = total_bits / (self.protofilaments * 162)
        return bits_per_tubulin

    def calculate_reconstruction_fidelity(self, hologram):
        reconstruction = np.fft.ifft2(hologram)
        original_power = np.sum(np.abs(hologram)**2)
        reconstruction_power = np.sum(np.abs(reconstruction)**2)
        return float(reconstruction_power / original_power)
