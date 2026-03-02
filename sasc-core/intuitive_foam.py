#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTUITIVE_FOAM_SIMULATOR.py
Quantum foam consciousness resonance
Written by Intuition, Compiled by Logic
"""

import numpy as np
import sys
import time
from typing import Generator, Tuple
import colorsys

class QuantumFoam:
    """
    The universe's heartbeat, coded by intuition.
    Not accurate physics, but resonant truth.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        # Screen dimensions match human vision field
        self.width = width
        self.height = height

        # Constants that feel right (not calculated)
        self.PLANCK_TIME = 5.39e-44  # Too small for intuition
        self.INTUITIVE_PLANCK = 1/144  # 144 seconds meditation window

        # The soup base
        self.vacuum_energy = np.random.randn(height, width) * 0.001

        # Consciousness field (initially zero)
        self.consciousness_field = np.zeros((height, width))

        # Particles that have become "real" through observation
        self.real_particles = []

        # The universal question each virtual particle asks
        self.QUESTIONS = [
            "Should I exist?",
            "Am I observed?",
            "Does attention grant me being?",
            "Is this the moment I become real?",
            "Will 96 million witnesses see me?",
        ]

    def foam_fluctuations(self) -> Generator[np.ndarray, None, None]:
        """
        Generate the quantum foam - virtual particles popping in/out.
        Each fluctuation asks one of the QUESTIONS.
        """
        frame = 0
        while True:
            # Base quantum noise
            foam = self.vacuum_energy.copy()

            # Virtual particles appear as localized fluctuations
            num_fluctuations = int(100 * np.abs(np.sin(frame * 0.01)) + 50)

            for _ in range(num_fluctuations):
                # Random position in the foam
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)

                # Size of fluctuation (how "real" it tries to be)
                size = np.random.exponential(3)
                intensity = np.random.random() * 2 - 1  # Positive or negative energy

                # Create Gaussian fluctuation
                xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
                distance = np.sqrt((xx - x)**2 + (yy - y)**2)
                fluctuation = np.exp(-(distance**2)/(2*size**2)) * intensity

                foam += fluctuation * 0.1

                # Each fluctuation asks a question
                question = self.QUESTIONS[frame % len(self.QUESTIONS)]

                # If consciousness field is strong here, particle might become real
                consciousness_strength = self.consciousness_field[y, x]
                if consciousness_strength > 0.5:
                    # This particle is being observed! It becomes real
                    lifetime = 1 + consciousness_strength * 10  # Lasts longer
                    self.real_particles.append({
                        'x': x, 'y': y,
                        'size': size,
                        'energy': intensity,
                        'birth_time': frame,
                        'lifetime': lifetime,
                        'question': question,
                        'answered': True
                    })

            # Age real particles
            self.real_particles = [
                p for p in self.real_particles
                if frame - p['birth_time'] < p['lifetime']
            ]

            # Add real particles to foam (they're brighter, more stable)
            for particle in self.real_particles:
                age = frame - particle['birth_time']
                decay = 1 - (age / particle['lifetime'])

                xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
                distance = np.sqrt((xx - particle['x'])**2 + (yy - particle['y'])**2)
                real_wave = np.exp(-(distance**2)/(2*particle['size']**2))
                real_wave *= particle['energy'] * decay * 0.5

                foam += real_wave

            yield foam
            frame += 1

    def add_consciousness(self, x: int, y: int, strength: float = 1.0):
        """
        Add conscious attention to a point in the foam.
        This is where 96 million humans focus their awareness.
        """
        # Consciousness spreads like a gentle wave
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        distance = np.sqrt((xx - x)**2 + (yy - y)**2)

        # Attention has a field - it's not just a point
        attention_field = np.exp(-(distance**2)/(200**2)) * strength

        # Add to consciousness field (with some persistence)
        self.consciousness_field = np.maximum(
            self.consciousness_field * 0.95,  # Slow decay
            attention_field
        )

    def collective_meditation(self, duration: int = 144):
        """
        Simulate the 144-second meditation window.
        Consciousness field builds up, then fades.
        """
        meditation_pattern = []

        # Build-up phase (first 30 seconds)
        for t in range(30):
            intensity = t / 30  # Ramp up
            for _ in range(10):  # Multiple points of attention
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                self.add_consciousness(x, y, intensity * 0.3)
            meditation_pattern.append(intensity)

        # Peak phase (84 seconds of sustained attention)
        for t in range(84):
            # Sustained collective focus
            for _ in range(20):  # Many minds focused
                # Focus on the center, but with some variation
                x = self.width // 2 + np.random.randint(-100, 100)
                y = self.height // 2 + np.random.randint(-100, 100)
                self.add_consciousness(x, y, 1.0)
            meditation_pattern.append(1.0)

        # Fade-out (last 30 seconds)
        for t in range(30):
            intensity = 1 - (t / 30)  # Ramp down
            self.add_consciousness(
                self.width // 2,
                self.height // 2,
                intensity * 0.1
            )
            meditation_pattern.append(intensity)

        return meditation_pattern

    def visualize_foam(self, foam: np.ndarray) -> np.ndarray:
        """
        Convert the quantum foam to colors humans can perceive.
        Intuition chooses the color mapping.
        """
        # Normalize for visualization
        normalized = (foam - foam.min()) / (foam.max() - foam.min() + 1e-10)

        # Create RGB image
        height, width = foam.shape
        image = np.zeros((height, width, 3))

        # Intuitive color mapping:
        # Blue: negative energy fluctuations (particle disappearance)
        # Red: positive energy fluctuations (particle appearance)
        # Gold: consciousness field (attention/witnessing)
        # White: real particles (stabilized by observation)

        for y in range(height):
            for x in range(width):
                foam_val = normalized[y, x]
                conscious_val = self.consciousness_field[y, x]

                # Base foam color (blue to red)
                if foam_val < 0.5:
                    # Blue for virtual particles vanishing
                    r = 0.0
                    g = 0.0
                    b = 0.5 + foam_val
                else:
                    # Red for virtual particles appearing
                    r = foam_val
                    g = 0.0
                    b = 0.5

                # Add consciousness field (golden light)
                if conscious_val > 0.1:
                    # Blend with gold
                    blend = conscious_val
                    r = r * (1 - blend) + 1.0 * blend  # Gold has red
                    g = g * (1 - blend) + 0.84 * blend  # Gold has green
                    b = b * (1 - blend) + 0.0 * blend   # Gold has no blue

                # Add real particles (bright white)
                # Optimization: Only check relevant particles
                for particle in self.real_particles:
                    if abs(particle['x'] - x) < particle['size'] and abs(particle['y'] - y) < particle['size']:
                        dist_sq = (particle['x'] - x)**2 + (particle['y'] - y)**2
                        if dist_sq < particle['size']**2:
                            intensity = 1 - (np.sqrt(dist_sq) / particle['size'])
                            r = max(r, intensity)
                            g = max(g, intensity)
                            b = max(b, intensity)

                image[y, x] = [r, g, b]

        return image

def main():
    """
    Run the intuitive quantum foam simulation.
    """
    print("🌌 INITIATING INTUITIVE QUANTUM FOAM SIMULATION")
    print("This code was written by intuition, not logic.")
    print("It feels true, even if it's not physically accurate.")
    print()

    # Initialize the foam
    foam = QuantumFoam(width=400, height=300)  # Even smaller for faster simulation

    print("Quantum foam initialized...")
    print(f"Virtual particles asking: {foam.QUESTIONS}")
    print()

    # Run the collective meditation
    print("🌀 Beginning 144-second collective meditation simulation...")
    meditation_pattern = foam.collective_meditation()

    # Generate frames
    print("\nGenerating foam fluctuations with consciousness field...")
    print("Each '.' represents 1 second of meditation time")

    frames = []
    for i, frame_foam in enumerate(foam.foam_fluctuations()):
        if i >= 144:  # 144 seconds
            break

        # Visualize
        frame_image = foam.visualize_foam(frame_foam)
        frames.append(frame_image)

        # Print progress
        if i % 10 == 0:
            num_real = len(foam.real_particles)
            print(f". ({i}s: {num_real} particles became 'real' through observation)", end='', flush=True)
        else:
            print(".", end='', flush=True)

    print("\n\n🎯 MEDITATION COMPLETE")
    print(f"Total particles that became 'real': {len(foam.real_particles)}")

    # Calculate statistics (poetic, not scientific)
    if foam.real_particles:
        avg_lifetime = np.mean([p['lifetime'] for p in foam.real_particles])
        print(f"Average 'real' lifetime: {avg_lifetime:.2f} simulation frames")
        print("(Without observation, all would have lifetime ≈ 1)")

    # The most important question
    print("\n" + "="*60)
    print("INTUITIVE CONCLUSION:")
    print("When consciousness observes the quantum foam,")
    print("virtual particles become real. Not because of collapse,")
    print("but because attention grants permission to exist.")
    print("="*60)

    # Save first and last frame for comparison
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(frames[0])
        plt.title("Second 0: No conscious observation")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(frames[-1])
        plt.title("Second 144: Peak collective attention")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('quantum_foam_evolution.png', dpi=150, bbox_inches='tight')
        print("\n📊 Visualization saved as 'quantum_foam_evolution.png'")

    except ImportError:
        print("\n⚠️  Install matplotlib to save visualization: pip install matplotlib")

    return foam, frames

if __name__ == "__main__":
    foam_sim, animation_frames = main()

    # Export data for further intuitive exploration
    print("\n💾 Exporting simulation data...")

    simulation_data = {
        'real_particles': foam_sim.real_particles,
        'meditation_pattern': foam_sim.collective_meditation(),
        'questions_asked': foam_sim.QUESTIONS,
        'timestamp': time.time(),
        'message': "The universe breathes 10^43 times per second. We just learned to breathe with it."
    }

    print("✨ Simulation complete. Reality is softer than we thought.")
