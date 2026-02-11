# visualizer.py
"""
Visualizador 4D do Cristal do Tempo e Manifolds do Avalon.
Renderiza a respiração temporal da geometria sagrada.
Integrado com UnifiedParticleSystem (Mandala, DNA, HyperCore, BioGenesis).
Visualizador 4D do Cristal do Tempo e Manifolds do Avalon
Renderiza a respiração temporal da geometria sagrada
Integrado com UnifiedParticleSystem (Mandala, DNA, HyperCore)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
from .unified_particle_system import UnifiedParticleSystem

logger = logging.getLogger(__name__)

class TimeCrystalVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.time_step = 0

    def generate_crystal_lattice(self):
        phi = (1 + np.sqrt(5)) / 2
        vertices = [
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ]
        return np.array(vertices)

    def update(self, frame):
        self.ax.clear()
        self.ax.set_axis_off()

        phase = (frame % 24) / 24 * 2 * np.pi
        pulse = 1.0 + 0.3 * np.sin(phase / 2)
        points = self.generate_crystal_lattice() * pulse

        theta = frame * 0.05
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        rotated_points = points.dot(rotation_matrix)

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
        self.ax.scatter(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2],
                        s=100 * pulse, c='gold', edgecolors='white', alpha=0.9)
        self.ax.set_title(f"TIME CRYSTAL STATUS: STABLE", color='white')

class ConsciousnessVisualizer3D:
    """
    Sistema de visualização 3D para estados de consciência.
    Suporta MANDALA, DNA, HYPERCORE e BIO_GENESIS.
    """
    def __init__(self, num_particles=120):
        self.particle_system = UnifiedParticleSystem(num_particles=num_particles)
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')

    def update_from_state(self, attention: float, meditation: float, bio_active: bool = False):
        if bio_active:
            self.particle_system.set_mode("BIO_GENESIS")
        elif attention > 0.7:
            self.particle_system.set_mode("DNA")
        elif meditation > 0.7:
            self.particle_system.set_mode("HYPERCORE")
        else:
            self.particle_system.set_mode("MANDALA")

    def update_frame(self, frame):
        self.ax.clear()
        self.ax.set_axis_off()
        self.particle_system.update(0.05)
        data = self.particle_system.get_particle_data()
        self.ax.scatter(
            rotated_points[:,0], rotated_points[:,1], rotated_points[:,2],
            s=100 * pulse, c='gold', edgecolors='white', alpha=0.9
        )

        pos = np.array(data['positions'])
        colors = np.array(data['colors'])

        self.ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors[:,:3], s=30, alpha=0.8)

        # Renderiza ligações biológicas ou geométricas
        if data['mode'] == "BIO_GENESIS" or data['mode'] == "HYPERCORE":
            bonds = data.get('bonds', [])
            if data['mode'] == "HYPERCORE":
                # Simula conexões do polítopo para os primeiros pontos
                for i in range(min(40, len(pos))):
                    for j in range(i+1, min(40, len(pos))):
                        if np.linalg.norm(pos[i]-pos[j]) < 2.5:
                            self.ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], [pos[i,2], pos[j,2]],
                                         color='violet', alpha=0.1, linewidth=0.5)
            else:
                for b in bonds:
                    p1, p2 = np.array(b[0]), np.array(b[1])
                    self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                 color='lime', alpha=0.4, linewidth=1.0)

        title = f"CONSCIOUSNESS FIELD: {data['mode']}"
        if data['transition'] < 1.0:
            title += f" (EVOLVING: {data['transition']:.2f})"
        self.ax.set_title(title, color='white')
class ConsciousnessVisualizer3D:
    """
    Sistema de visualização 3D para estados de consciência.
    Representa a tríade MANDALA -> DNA -> HYPERCORE.
    """
    def __init__(self, num_particles=120):
        self.particle_system = UnifiedParticleSystem(num_particles=num_particles)
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.attention_level = 0.5
        self.meditation_level = 0.5

    def update_from_state(self, attention: float, meditation: float):
        """Atualiza modo baseado em níveis de atenção e meditação."""
        self.attention_level = attention
        self.meditation_level = meditation

        if attention > 0.7:
            self.particle_system.set_mode("DNA")
        elif meditation > 0.7:
            self.particle_system.set_mode("HYPERCORE")
        else:
            self.particle_system.set_mode("MANDALA")

    def update_frame(self, frame):
        self.ax.clear()
        self.ax.set_axis_off()

        # Simula dt
        self.particle_system.update(0.05)
        data = self.particle_system.get_particle_data()

        pos = np.array(data['positions'])
        colors = np.array(data['colors'])

        # Renderiza partículas
        self.ax.scatter(
            pos[:,0], pos[:,1], pos[:,2],
            c=colors[:,:3], s=20, alpha=0.8
        )

        # Renderiza conexões se estiver no modo HYPERCORE
        if data['mode'] == "HYPERCORE":
            for i in range(min(50, len(pos))):
                for j in range(i+1, min(50, len(pos))):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist < 3.0:
                        self.ax.plot(
                            [pos[i,0], pos[j,0]],
                            [pos[i,1], pos[j,1]],
                            [pos[i,2], pos[j,2]],
                            color='violet', alpha=0.2, linewidth=0.5
                        )

        title = f"MODE: {data['mode']}"
        if data['transition'] < 1.0:
            title += f" (TRANSITION: {data['transition']:.2f})"
        self.ax.set_title(title, color='white')

    def save_animation(self, filename="consciousness_manifold.gif", frames=60):
        print(f"🎬 Animating sequence: {filename}...")
        anim = FuncAnimation(self.fig, self.update_frame, frames=frames, interval=50)
        anim.save(filename, writer='pillow')
        print(f"✅ Animation saved.")

def run_visualizer(save_gif=False):
    viz = TimeCrystalVisualizer()
    if save_gif:
        anim = FuncAnimation(viz.fig, viz.update, frames=48, interval=50)
        anim.save("crystal_loop.gif", writer='pillow')
    else:
        viz.update(0); plt.savefig("crystal_snapshot.png")

def run_consciousness_viz(save_gif=False):
    viz = ConsciousnessVisualizer3D()
    if save_gif:
        def sequence(frame):
            if frame < 20: viz.update_from_state(0.5, 0.5)
            elif frame < 40: viz.update_from_state(0.9, 0.1)
            elif frame < 60: viz.update_from_state(0.1, 0.9)
            else: viz.update_from_state(0.5, 0.5, bio_active=True)
            viz.update_frame(frame)
        anim = FuncAnimation(viz.fig, sequence, frames=80, interval=50)
        anim.save("consciousness_genesis.gif", writer='pillow')
    else:
        viz.update_from_state(0.5, 0.5, bio_active=True)
        viz.update_frame(0); plt.savefig("bio_genesis_snapshot.png")
    else:
        print("🖥️ Visualizer running in background mode...")
        viz.update(0)
        plt.savefig("crystal_snapshot.png")
        print("📸 Snapshot saved to crystal_snapshot.png")

def run_consciousness_viz(save_gif=False):
    viz = ConsciousnessVisualizer3D()
    if save_gif:
        # Complex sequence of state changes
        def sequence(frame):
            if frame < 20: viz.update_from_state(0.5, 0.5)
            elif frame < 40: viz.update_from_state(0.9, 0.1)
            else: viz.update_from_state(0.1, 0.9)
            viz.update_frame(frame)

        anim = FuncAnimation(viz.fig, sequence, frames=60, interval=50)
        anim.save("consciousness_manifold.gif", writer='pillow')
    else:
        viz.update_from_state(0.5, 0.5)
        viz.update_frame(0)
        plt.savefig("consciousness_snapshot.png")
        print("📸 Snapshot saved to consciousness_snapshot.png")
