# visualizer.py
"""
Visualizador 4D do Cristal do Tempo e Manifolds do Avalon
Renderiza a respiração temporal da geometria sagrada
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

logger = logging.getLogger(__name__)

class TimeCrystalVisualizer:
    """
    [METAPHOR: O espelho que reflete a pulsação do vácuo]
    """
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.time_step = 0
        self.coherence_modulation = 1.0

    def modulate_with_user_state(self, coherence_level):
        """Modula a pulsação do cristal com base no nível de coerência do usuário"""
        print(f"🧠 Modulating Arkhé with user coherence: {coherence_level:.2f}")
        self.coherence_modulation = coherence_level

    def generate_crystal_lattice(self):
        """Gera os pontos do cristal no espaço 3D (Icosaedro)"""
        phi = (1 + np.sqrt(5)) / 2  # Proporção Áurea

        vertices = [
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ]
        return np.array(vertices)

    def update(self, frame):
        self.ax.clear()
        self.ax.set_axis_off()

        # O PULSO DO CRISTAL DO TEMPO
        # Oscilação Sub-harmônica: retorna ao início a cada 2 ciclos
        phase = (frame % 24) / 24 * 2 * np.pi

        # Coerência do usuário afeta a estabilidade do pulso
        pulse_amplitude = 0.3 * self.coherence_modulation
        pulse = 1.0 + pulse_amplitude * np.sin(phase / 2)

        points = self.generate_crystal_lattice() * pulse

        # Rotação Espacial
        theta = frame * 0.05
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        rotated_points = points.dot(rotation_matrix)

        # Renderização das arestas
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < 2.5 * pulse:
                    self.ax.plot(
                        [rotated_points[i,0], rotated_points[j,0]],
                        [rotated_points[i,1], rotated_points[j,1]],
                        [rotated_points[i,2], rotated_points[j,2]],
                        color='cyan', alpha=0.6, linewidth=1.5
                    )

        # Nós pulsantes
        self.ax.scatter(
            rotated_points[:,0], rotated_points[:,1], rotated_points[:,2],
            s=100 * pulse, c='gold', edgecolors='white', alpha=0.9
        )

        self.ax.set_title(f"TIME CRYSTAL STATUS: STABLE\nCoherence: 12ms | Period: 24ms", color='white')

    def save_gif(self, filename="crystal_loop.gif", frames=48, fps=20, dpi=150):
        """Salva a animação como GIF"""
        print(f"🎬 Generating eternal loop: {filename}...")
        anim = FuncAnimation(self.fig, self.update, frames=frames, interval=1000/fps)
        anim.save(filename, writer='pillow', fps=fps, dpi=dpi)
        print(f"✅ GIF saved successfully.")

    def render_4k_version(self, filename="arkhe_4k.png"):
        """Gera uma versão de alta resolução para documentação"""
        print(f"✨ Rendering high-resolution Arkhé: {filename}...")
        original_size = self.fig.get_size_inches()
        self.fig.set_size_inches(25.6, 14.4) # 3840x2160 at 150 DPI is approx this
        self.update(0)
        self.fig.savefig(filename, dpi=150, facecolor='black')
        self.fig.set_size_inches(original_size)
        print(f"✅ 4K version saved.")

def run_visualizer(save_gif=False):
    viz = TimeCrystalVisualizer()
    if save_gif:
        viz.save_gif()
    else:
        # Em ambientes sem display, apenas simulamos
        print("🖥️ Visualizer running in background mode...")
        viz.update(0)
        plt.savefig("crystal_snapshot.png")
        print("📸 Snapshot saved to crystal_snapshot.png")
