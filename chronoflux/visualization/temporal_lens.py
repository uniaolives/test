"""
Temporal Lens image processing logic.
Simulates 'Temporal Echoes' (motion ghosts) based on Chronoflux vorticity.
"""
import cv2
import numpy as np

class TemporalLens:
    """
    Implements the 'Lente de Eventos' visualization.
    Uses frame accumulation to create ghosts where vorticity is high.
    """
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.frame_buffer = []

    def process_frame(self, frame, omega):
        """
        Applies temporal echo effect to the frame.
        Persistence (ghosting) scales with omega.
        """
        self.frame_buffer.append(frame.astype(np.float32))
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

        # Base persistence: scales with omega (0.0 to 1.0+)
        # higher omega -> more weight to old frames
        persistence = np.clip(omega * 0.8, 0.1, 0.95)

        # Calculate weighted average (echo)
        echo = np.zeros_like(frame, dtype=np.float32)
        weights = [persistence ** i for i in range(len(self.frame_buffer))][::-1]
        total_weight = sum(weights)

        for f, w in zip(self.frame_buffer, weights):
            echo += f * (w / total_weight)

        # Highlight areas of movement (diff) with high omega
        if len(self.frame_buffer) > 1:
            diff = cv2.absdiff(frame.astype(np.float32), self.frame_buffer[-2])
            # Distortion effect: amplify movement if omega is high
            echo += diff * omega * 0.5

        return np.clip(echo, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    lens = TemporalLens()
    print("Temporal Lens Logic Initialized.")
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = lens.process_frame(dummy_frame, 0.9)
    print(f"Frame processed. Shape: {res.shape}")
