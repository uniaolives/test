"""
Anisotropic Cone Visualization (χ_DUALIS).
Visualizes the Semi-Dirac dispersion relation: E ∝ p² (x) and E ∝ |p| (y).
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_semidirac_dispersion(filename='semidirac_dispersion.png'):
    px = np.linspace(-1, 1, 100)
    py = np.linspace(-1, 1, 100)
    PX, PY = np.meshgrid(px, py)

    # E = sqrt((px²)² + py²) - Simplified Semi-Dirac
    E = np.sqrt((PX**2)**2 + PY**2)

    fig = plt.figure(figsize=(10, 8), facecolor='#0d0d0d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d0d0d')

    # Surface plot
    surf = ax.plot_surface(PX, PY, E, cmap='magma', edgecolor='none', alpha=0.8)

    # Customizing axes
    ax.set_title('χ_DUALIS: Semi-Dirac Dispersion (ZrSiS)', color='white', fontsize=14)
    ax.set_xlabel('p_x (Massive Axis)', color='white')
    ax.set_ylabel('p_y (Massless Axis)', color='white')
    ax.set_zlabel('Energy (E)', color='white')

    # White grid and ticks
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0.1)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0.1)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0.1)

    plt.savefig(filename, dpi=150)
    print(f"Visualização salva em {filename}")

if __name__ == "__main__":
    plot_semidirac_dispersion()
