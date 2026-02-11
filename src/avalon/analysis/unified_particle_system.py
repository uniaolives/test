"""
Unified Particle System for Consciousness Representation.
Integrates Mandala (Order), DNA (Evolution), Hyper-Core (4D Unification), and BIO_GENESIS (Biological Emergence).
"""

import math
import numpy as np
import itertools
from typing import Dict, List, Tuple, Any, Optional
from ..core.bio_arkhe import BioAgent, MorphogeneticField, ArkheGenome

def get_mandala_pos(index, total_particles, time_pulse):
    """Generates positions for a Mandala pattern."""
    n_rings = 5
    particles_per_ring = total_particles // n_rings
    if particles_per_ring == 0: particles_per_ring = 1
    ring_idx = index // particles_per_ring
    particle_in_ring_idx = index % particles_per_ring
    radius = (ring_idx + 1) * 2.0
    angle = (particle_in_ring_idx / particles_per_ring) * 2 * np.pi + time_pulse * 0.1
    z = 0.5 * math.sin(time_pulse + ring_idx)
    return np.array([radius * math.cos(angle), radius * math.sin(angle), z])

def get_dna_pos(index, total_particles, time_pulse):
    """Generates positions for a DNA double helix pattern."""
    strand = index % 2
    z_step = 0.2
    angle_step = 0.5
    t = (index // 2) * z_step - (total_particles // 4) * z_step
    angle = (index // 2) * angle_step + strand * np.pi + time_pulse * 0.5
    radius = 2.0
    return np.array([radius * math.cos(angle), radius * math.sin(angle), t])

def get_hypercore_pos(index, total_particles, time_pulse, rotation_4d=True):
    """Gera posições 3D projetadas a partir do Hecatonicosachoron 4D."""
    vertices_4d = []
    phi = (1 + math.sqrt(5)) / 2
    ones = [1, -1]
    for i in range(4):
        for sign in ones:
            v = [0, 0, 0, 0]; v[i] = sign
            vertices_4d.append(v)
    half = 0.5
    for sx in ones:
        for sy in ones:
            for sz in ones:
                for sw in ones:
                    if ((sx < 0) + (sy < 0) + (sz < 0) + (sw < 0)) % 2 == 0:
                        vertices_4d.append([sx*half, sy*half, sz*half, sw*half])
    if len(vertices_4d) < 120:
        n = 120 - len(vertices_4d)
        indices = np.arange(0, n, dtype=float) + 0.5
        phi_inv = (np.sqrt(5) - 1) / 2
        for i in indices:
            phi1 = 2 * np.pi * i * phi_inv
            phi2 = 2 * np.pi * i * (phi_inv**2)
            z = 1 - (2 * i) / n
            r = np.sqrt(max(0, 1 - z*z))
            vertices_4d.append([r * np.cos(phi1), r * np.sin(phi1), z * np.cos(phi2), z * np.sin(phi2)])
    vertex_idx = index % len(vertices_4d)
    point_4d = np.array(vertices_4d[vertex_idx])
    if rotation_4d:
        a1, a2 = time_pulse * 0.2, time_pulse * 0.15
        c1, s1, c2, s2 = math.cos(a1), math.sin(a1), math.cos(a2), math.sin(a2)
        x, y, z, w = point_4d
        x_new, y_new = x * c1 - y * s1, x * s1 + y * c1
        z_new, w_new = z * c2 - w * s2, z * s2 + w * c2
        point_4d = np.array([x_new, y_new, z_new, w_new])
    x, y, z, w = point_4d
    if abs(1 - w) < 0.001: w = 0.999
    scale = 1.0 / (1.0 - w)
    pulse = 1.0 + 0.1 * math.sin(time_pulse * 3 + index * 0.1)
    return np.array([x * scale * pulse, y * scale * pulse, z * scale * pulse])

class UnifiedParticleSystem:
    """Sistema de partículas com modos Top-Down e Emergência Biológica."""

    def __init__(self, num_particles=120):
        self.num_particles = num_particles
        self.time = 0.0
        self.current_mode = "MANDALA"
        self.target_mode = "MANDALA"
        self.transition_progress = 1.0
        self.transition_speed = 0.02
        self.quantum_jitter = 0.01

        # BIO-ARKHE Components
        self.field = MorphogeneticField()
        self.agents: Dict[int, BioAgent] = {}

        # Partículas unificadas
        self.particles = []
        for i in range(num_particles):
            # Genoma padrão
            genome = ArkheGenome(C=0.6, I=0.5, E=0.4, F=0.5)
            pos = np.random.uniform(-5, 5, 3)
            agent = BioAgent(i, pos, genome)
            self.agents[i] = agent

            self.particles.append({
                'index': i,
                'pos': pos,
                'target_pos': pos.copy(),
                'color': [1.0, 1.0, 1.0, 1.0],
                'size': 1.0,
                'energy': 0.5 + 0.5 * math.sin(i * 0.1)
            })

    def set_mode(self, new_mode):
        if new_mode in ["MANDALA", "DNA", "HYPERCORE", "BIO_GENESIS"]:
            self.target_mode = new_mode
            self.transition_progress = 0.0

    def update(self, dt):
        self.time += dt

        # Transição de modos
        if self.current_mode != self.target_mode:
            self.transition_progress += self.transition_speed
            if self.transition_progress >= 1.0:
                self.transition_progress = 1.0
                self.current_mode = self.target_mode

        # Modo BIO_GENESIS utiliza agentes autônomos
        if self.target_mode == "BIO_GENESIS":
            self.field.update_field(list(self.agents.values()))
            for i, agent in self.agents.items():
                agent.sense_and_act(self.field, self.agents)
                self.particles[i]['pos'] = agent.position
                self.update_particle_color(self.particles[i])
        else:
            # Modos Top-Down
            for p in self.particles:
                idx = p['index']
                if self.target_mode == "MANDALA":
                    target = get_mandala_pos(idx, self.num_particles, self.time)
                elif self.target_mode == "DNA":
                    target = get_dna_pos(idx, self.num_particles, self.time)
                else: # HYPERCORE
                    target = get_hypercore_pos(idx, self.num_particles, self.time)

                # Interpolação de transição
                if self.transition_progress < 1.0:
                    current_pos = p['pos'] # Simplificado
                    t = self.transition_progress
                    smooth_t = t * t * (3 - 2 * t)
                    p['target_pos'] = current_pos * (1 - smooth_t) + target * smooth_t
                else:
                    p['target_pos'] = target

                p['pos'] = p['pos'] * 0.9 + p['target_pos'] * 0.1
                self.update_particle_color(p)
                p['pos'] += np.random.normal(0, self.quantum_jitter, 3)
                # Sincroniza posição do agente
                self.agents[idx].position = p['pos']

    def update_particle_color(self, particle):
        if self.target_mode == "MANDALA":
            h, s, v = 0.12, 0.8, 0.7 + 0.3 * particle['energy']
        elif self.target_mode == "DNA":
            h, s, v = 0.5, 0.9, 0.6 + 0.4 * math.sin(self.time + particle['index'] * 0.1)
        elif self.target_mode == "HYPERCORE":
            h, s, v = 0.8, 0.7, 0.5 + 0.5 * math.sin(self.time * 2 + particle['index'] * 0.05)
        else: # BIO_GENESIS
            # Cor baseada em conexões
            n_neighbors = len(self.agents[particle['index']].neighbors)
            h = 0.3 + 0.1 * n_neighbors # Verde para agentes conectados
            s, v = 0.8, 0.6 + 0.4 * particle['energy']

        particle['color'] = self.hsv_to_rgb(h, s, v)

    def hsv_to_rgb(self, h, s, v):
        i = int(h * 6.); f = (h * 6.) - i
        p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
        i %= 6
        if i == 0: return [v, t, p, 1.0]
        if i == 1: return [q, v, p, 1.0]
        if i == 2: return [p, v, t, 1.0]
        if i == 3: return [p, q, v, 1.0]
        if i == 4: return [t, p, v, 1.0]
        if i == 5: return [v, p, q, 1.0]
        return [0, 0, 0, 1.0]

    def get_particle_data(self):
        bonds = []
        if self.target_mode == "BIO_GENESIS":
            for i, agent in self.agents.items():
                for nid in agent.neighbors:
                    if i < nid: # Evita duplicatas
                        bonds.append((agent.position.tolist(), self.agents[nid].position.tolist()))

        return {
            'positions': [p['pos'].tolist() for p in self.particles],
            'colors': [p['color'] for p in self.particles],
            'sizes': [p['size'] for p in self.particles],
            'mode': self.target_mode,
            'transition': self.transition_progress,
            'bonds': bonds
        }
