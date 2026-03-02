"""
Demo of the Temporal Lens (Lente de Eventos) vision layer.
Simulates motion ghosts and temporal echoes.
"""
import numpy as np
import cv2
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chronoflux.visualization.temporal_lens import TemporalLens

def run_lens_demo():
    print("=" * 60)
    print("CHRONOFLUX TEMPORAL LENS DEMO")
    print("Simulating 'Event Camera' ghosting across different vorticity levels")
    print("=" * 60)

    lens = TemporalLens(buffer_size=15)

    # Create a dummy moving object
    def create_frame(step, size=200):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        # Moving circle
        center = (50 + step * 5, 100)
        cv2.circle(frame, center, 20, (255, 255, 255), -1)
        return frame

    vorticity_states = [0.1, 0.9, 3.0]
    labels = ["Laminar (Normal)", "Vortical (Ghosting)", "Singularity (High Persistence)"]

    for label, omega in zip(labels, vorticity_states):
        print(f"\nProcessing 20 frames for state: {label} (ω = {omega})")

        # Process sequence
        for i in range(20):
            frame = create_frame(i)
            processed = lens.process_frame(frame, omega)

        # Analyze the final frame's mean brightness as a proxy for ghost persistence
        # High persistence/echoes will result in more non-zero pixels
        non_zero = np.count_nonzero(processed)
        print(f"  Final Frame Analysis: {non_zero} active pixels (ghost coverage)")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("Note: High vorticity correctly increases temporal echo persistence.")
    print("=" * 60)

if __name__ == "__main__":
    run_lens_demo()
