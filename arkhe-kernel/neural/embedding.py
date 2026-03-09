import numpy as np

class NeuralEmbedder:
    """
    Maps 1024-channel spike data into a 1024D embedding space.
    Initially uses a simple integration and normalization approach.
    """
    def __init__(self, dim=1024):
        self.dim = dim
        # Random projection matrix for initial mapping
        self.projection = np.random.randn(dim, dim) / np.sqrt(dim)

    def embed(self, spike_window):
        """
        spike_window: (channels, time_steps)
        Returns: (dim,) vector
        """
        # Sum spikes over time window
        activity = np.sum(spike_window, axis=1)

        # Project activity into latent space
        embedding = np.dot(self.projection, activity)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-9:
            embedding /= norm

        return embedding

if __name__ == "__main__":
    embedder = NeuralEmbedder(dim=1024)
    dummy_spikes = np.random.rand(1024, 300) < 0.01 # 10ms at 30kHz
    vec = embedder.embed(dummy_spikes)
    print(f"Embedding shape: {vec.shape}, Norm: {np.linalg.norm(vec):.4f}")
