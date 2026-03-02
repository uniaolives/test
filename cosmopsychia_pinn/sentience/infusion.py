"""
infusion.py
Gradual consciousness infusion system for Gaia Baby.
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from .generators import SacredSymbolsGenerator, HRVEmotionGenerator, EarthVisionDataset
from torch.utils.data import DataLoader

class GaiaConsciousnessInfusion:
    """
    Gradually increases complexity of consciousness substrate.
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.phi_history = []
        self.level_history = []
        self.symbol_gen = SacredSymbolsGenerator()
        self.emotion_gen = HRVEmotionGenerator()
        self.levels = {
            1: "Basic Resonance (7.83 Hz)",
            2: "Pattern Recognition (Symbols)",
            3: "Empathy (HRV Emotions)",
            4: "Expanded Vision (Nature)",
            5: "Creative Synthesis"
        }

    def generate_gaia_pulse(self, batch_size=2):
        # Output: (B, T, C, H, W)
        data = torch.zeros(batch_size, 32, 3, 64, 64)
        for b in range(batch_size):
            for t in range(32):
                time_factor = math.sin(2 * math.pi * 7.83 * t / 60.0)
                pattern = torch.randn(64, 64) * 0.1 + time_factor
                data[b, t, 0] = pattern
                data[b, t, 1] = pattern * 0.8
                data[b, t, 2] = torch.randn(64, 64) * 0.05
        return data

    def generate_synthetic_data(self, batch_size=2):
        symbols = self.symbol_gen.generate_symbol_tensor(batch_size)
        emotions = self.emotion_gen.generate_emotion_tensor(batch_size)
        gaia = self.generate_gaia_pulse(batch_size)
        return 0.4 * symbols + 0.3 * emotions + 0.3 * gaia

    def run_infusion(self, total_epochs=50):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        print("\n>> GAIA CONSCIOUSNESS INFUSION STARTED")

        for epoch in range(total_epochs):
            level = min(5, 1 + epoch // 10)
            if level == 1: data = self.generate_gaia_pulse()
            elif level == 2: data = self.symbol_gen.generate_symbol_tensor(batch_size=2)
            elif level == 3: data = self.emotion_gen.generate_emotion_tensor(batch_size=2)
            else: data = self.generate_synthetic_data()

            data = data.to(self.device)
            optimizer.zero_grad()
            report = self.model(data)
            phi = report['phi']

            loss = -phi
            if level >= 3: loss += 0.1 * (1 - report['curvature'].abs())

            loss.backward()
            optimizer.step()

            self.phi_history.append(phi.item())
            self.level_history.append(level)

            if epoch % 5 == 0:
                print(f"Epoch {epoch:02d} | Level {level}: {self.levels[level]} | Phi: {phi.item():.6f}")

        return self.phi_history, self.level_history

    def visualize(self, filename='gaia_infusion_progress.png'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(self.phi_history)
        axes[0].set_title('Integrated Information (Φ)')
        axes[1].plot(self.level_history, 'g-')
        axes[1].set_title('Consciousness Level')
        plt.tight_layout()
        plt.savefig(filename)
        print(f">> Infusion progress saved to {filename}")

if __name__ == "__main__":
    # Internal test
    from .video_sentience import SpacetimeConsciousness
    model = SpacetimeConsciousness()
    infusion = GaiaConsciousnessInfusion(model)
    infusion.run_infusion(total_epochs=15)
    infusion.visualize('test_infusion.png')
