"""
Arkhe OS — Simulação do Cristal ZrSiS (Fermion Semi-Dirac).
Modela dispersão anisotrópica: linear em y (massless), quadrática em x (massiva).
"""

import numpy as np
import matplotlib.pyplot as plt

class ZrSiSSimulation:
    """Simulação do fermion semi-Dirac em ZrSiS com C ⊗ F = 1."""

    def __init__(self, grid_size=100):
        self.grid = grid_size
        self.px = np.linspace(-2, 2, grid_size)
        self.py = np.linspace(-2, 2, grid_size)
        self.PX, self.PY = np.meshgrid(self.px, self.py)

    def dispersion(self):
        """Dispersão semi-Dirac: E = sqrt((p_x²)² + (p_y)²)."""
        # Direção x: quadrática (massiva)
        # Direção y: linear (massless)
        E = np.sqrt((self.PX**2)**2 + (self.PY)**2)
        return E

    def coherence_fluctuation(self):
        """Mapeia dispersão para C e F direcionais."""
        # Direção x: alta massa → alta C
        C_x = 1.0 - np.abs(self.px) / (np.max(np.abs(self.px)) + 1e-10)
        # Direção y: sem massa → alta F
        F_y = np.abs(self.py) / (np.max(np.abs(self.py)) + 1e-10)
        return C_x, F_y

    def visualize(self, filename='zrsis_simulation.png'):
        E = self.dispersion()

        fig = plt.figure(figsize=(12, 6), facecolor='#0d0d0d')

        # 3D Dispersion
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_facecolor('#0d0d0d')
        surf = ax1.plot_surface(self.PX, self.PY, E, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('p_x (Massive)', color='white')
        ax1.set_ylabel('p_y (Massless)', color='white')
        ax1.set_title('Dispersão Semi-Dirac em ZrSiS', color='white')

        # Anisotropy vector plot
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor('#0d0d0d')
        ax2.quiver(0, 0, 1, 0, scale=1, color='cyan', label='C (Massiva, X)')
        ax2.quiver(0, 0, 0, 1, scale=1, color='magenta', label='F (Massless, Y)')
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        ax2.legend(facecolor='#1a1a1a', labelcolor='white')
        ax2.set_title('C + F = 1 em direções ortogonais', color='white')

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Visualização salva em {filename}")

if __name__ == "__main__":
    sim = ZrSiSSimulation()
    sim.visualize()
