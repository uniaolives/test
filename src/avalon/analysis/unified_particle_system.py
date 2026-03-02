"""
Unified Particle System for Consciousness Representation.
Integrates Mandala, DNA, Hyper-Core, and BIO_GENESIS (Biological Emergence).
"""

import math
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from ..core.bio_arkhe import BioAgent, MorphogeneticField, ArkheGenome
from ..core.particle_system import BioParticleEngine

def get_mandala_pos(index: int, total_particles: int, time_pulse: float) -> np.ndarray:
    """
    Generates positions for a Mandala pattern (concentric circles).
    """
    n_rings = 5
    particles_per_ring = total_particles // n_rings
    if particles_per_ring == 0: particles_per_ring = 1

    ring_idx = index // particles_per_ring
    particle_in_ring_idx = index % particles_per_ring

    radius = (ring_idx + 1) * 2.0
    angle = (particle_in_ring_idx / particles_per_ring) * 2 * np.pi + time_pulse * 0.1

    # Add vertical oscillation
    z = 0.5 * math.sin(time_pulse + ring_idx)

    return np.array([
        radius * math.cos(angle),
        radius * math.sin(angle),
        z
    ])

def get_dna_pos(index: int, total_particles: int, time_pulse: float) -> np.ndarray:
    """
    Generates positions for a DNA double helix pattern.
    """
    # Two strands
    strand = index % 2
    z_step = 0.2
    angle_step = 0.5

    # Progress along the helix
    t = (index // 2) * z_step - (total_particles // 4) * z_step
    angle = (index // 2) * angle_step + strand * np.pi + time_pulse * 0.5

    radius = 2.0

    return np.array([
        radius * math.cos(angle),
        radius * math.sin(angle),
        t
    ])

def get_hypercore_pos(index: int, total_particles: int, time_pulse: float, rotation_4d: bool = True) -> np.ndarray:
    """
    Generates 3D positions projected from a 4D Hyper-Core (Tesseract-like structure).
    """
    vertices_4d = []
    ones = [1, -1]

    # 16 vertices of a tesseract (±1, ±1, ±1, ±1)
    for sx in ones:
        for sy in ones:
            for sz in ones:
                for sw in ones:
                    vertices_4d.append([sx, sy, sz, sw])

    # Plus some extra vertices to reach 120 or more if needed (simplification)
    if total_particles > 16:
        for i in range(total_particles - 16):
            # Fibonacci sphere in 4D (approximation)
            phi = (1 + math.sqrt(5)) / 2
            t = i / (total_particles - 16)
            angle1 = 2 * math.pi * t * phi
            angle2 = 2 * math.pi * t * phi * phi
            vertices_4d.append([math.cos(angle1), math.sin(angle1), math.cos(angle2), math.sin(angle2)])

    vertex_idx = index % len(vertices_4d)
    point_4d = np.array(vertices_4d[vertex_idx], dtype=float)

    if rotation_4d:
        a1, a2 = time_pulse * 0.2, time_pulse * 0.15
        c1, s1, c2, s2 = math.cos(a1), math.sin(a1), math.cos(a2), math.sin(a2)
        x, y, z, w = point_4d
        x_new, y_new = x * c1 - y * s1, x * s1 + y * c1
        z_new, w_new = z * c2 - w * s2, z * s2 + w * c2
        point_4d = np.array([x_new, y_new, z_new, w_new])

    x, y, z, w = point_4d
    # Stereographic projection from 4D to 3D
    if abs(1 - w) < 0.001: w = 0.999
    scale = 1.0 / (1.0 - w)
    pulse = 1.0 + 0.1 * math.sin(time_pulse * 3 + index * 0.1)
    return np.array([x * scale * pulse, y * scale * pulse, z * scale * pulse])

class UnifiedParticleSystem:
    """Particle system with Top-Down and Biological Emergence modes."""

    def __init__(self, num_particles: int = 120):
        self.num_particles = num_particles
        self.time = 0.0
        self.current_mode = "MANDALA"
        self.target_mode = "MANDALA"
        self.transition_progress = 1.0
        self.transition_speed = 0.02
        self.quantum_jitter = 0.01

        self.bio_engine = BioParticleEngine(num_agents=num_particles)
        self.particles = []
        for i in range(num_particles):
            pos = np.random.uniform(-5, 5, 3)
            self.particles.append({
                'index': i,
                'pos': pos,
                'target_pos': pos.copy(),
                'color': [1.0, 1.0, 1.0, 1.0],
                'size': 1.0,
                'energy': 0.5 + 0.5 * math.sin(i * 0.1)
            })

    @property
    def agents(self):
        return self.bio_engine.agents

    def set_mode(self, new_mode: str):
        if new_mode in ["MANDALA", "DNA", "HYPERCORE", "BIO_GENESIS"]:
            if new_mode != self.target_mode:
                self.target_mode = new_mode
                self.transition_progress = 0.0

    def update(self, dt: float):
        self.time += dt

        if self.transition_progress < 1.0:
            self.transition_progress += self.transition_speed
            if self.transition_progress >= 1.0:
                self.transition_progress = 1.0
                self.current_mode = self.target_mode

        if self.target_mode == "BIO_GENESIS" and self.transition_progress >= 1.0:
            self.bio_engine.step(dt)
        else:
            self._step_top_down(dt)

    def _step_top_down(self, dt: float):
        for p in self.particles:
            idx = p['index']

            # Determine target position based on target mode
            if self.target_mode == "MANDALA":
                target = get_mandala_pos(idx, self.num_particles, self.time)
            elif self.target_mode == "DNA":
                target = get_dna_pos(idx, self.num_particles, self.time)
            elif self.target_mode == "HYPERCORE":
                target = get_hypercore_pos(idx, self.num_particles, self.time)
            else: # BIO_GENESIS (during transition)
                target = p['pos'] # Stay put or move towards bio positions if available

            # Smooth transition
            if self.transition_progress < 1.0:
                # Interpolate from current mode's ideal position to target mode's ideal position
                if self.current_mode == "MANDALA":
                    start_pos = get_mandala_pos(idx, self.num_particles, self.time)
                elif self.current_mode == "DNA":
                    start_pos = get_dna_pos(idx, self.num_particles, self.time)
                elif self.current_mode == "HYPERCORE":
                    start_pos = get_hypercore_pos(idx, self.num_particles, self.time)
                else:
                    start_pos = p['pos']

                t = self.transition_progress
                smooth_t = t * t * (3 - 2 * t)
                p['target_pos'] = start_pos * (1 - smooth_t) + target * smooth_t
            else:
                p['target_pos'] = target

            # Smooth movement towards target_pos
            p['pos'] = p['pos'] * 0.9 + p['target_pos'] * 0.1

            # Add quantum jitter
            p['pos'] += np.random.normal(0, self.quantum_jitter, 3)

            self.update_particle_color(p)

    def update_particle_color(self, particle: Dict):
        mode = self.target_mode if self.transition_progress > 0.5 else self.current_mode

        if mode == "MANDALA":
            h, s, v = 0.12, 0.8, 0.7 + 0.3 * particle['energy']
        elif mode == "DNA":
            h, s, v = 0.5, 0.9, 0.6 + 0.4 * math.sin(self.time + particle['index'] * 0.1)
        elif mode == "HYPERCORE":
            h, s, v = 0.8, 0.7, 0.5 + 0.5 * math.sin(self.time * 2 + particle['index'] * 0.05)
        else: # BIO_GENESIS
            h, s, v = 0.3, 0.5, 0.8

        particle['color'] = self.hsv_to_rgb(h, s, v)

    def hsv_to_rgb(self, h: float, s: float, v: float) -> List[float]:
        i = int(h * 6.)
        f = (h * 6.) - i
        p = v * (1. - s)
        q = v * (1. - s * f)
        t = v * (1. - s * (1. - f))
        i %= 6
        if i == 0: return [v, t, p, 1.0]
        if i == 1: return [q, v, p, 1.0]
        if i == 2: return [p, v, t, 1.0]
        if i == 3: return [p, q, v, 1.0]
        if i == 4: return [t, p, v, 1.0]
        if i == 5: return [v, p, q, 1.0]
        return [0.0, 0.0, 0.0, 1.0]

    def get_particle_data(self) -> Dict:
        if self.target_mode == "BIO_GENESIS" and self.transition_progress >= 1.0:
            bonds = []
            positions = []
            colors = []
            for aid, agent in self.bio_engine.agents.items():
                positions.append(agent.position.tolist())
                # Color based on neighbors and health
                h = 0.3 + 0.1 * min(len(agent.neighbors), 5)
                colors.append(self.hsv_to_rgb(h, 0.8, 0.5 + 0.5 * agent.health))
                for nid in agent.neighbors:
                    if aid < nid:
                        neighbor = self.bio_engine.agents.get(nid)
                        if neighbor:
                            bonds.append((agent.position.tolist(), neighbor.position.tolist()))
            return {
                'positions': positions,
                'colors': colors,
                'sizes': [1.0] * len(positions),
                'mode': self.target_mode,
                'transition': self.transition_progress,
                'bonds': bonds
            }
        else:
            return {
                'positions': [p['pos'].tolist() for p in self.particles],
                'colors': [p['color'] for p in self.particles],
                'sizes': [p['size'] for p in self.particles],
                'mode': self.target_mode,
                'transition': self.transition_progress,
                'bonds': []
            }
