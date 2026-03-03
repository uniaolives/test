"""
safe_infusion.py
Safe emotional infusion for Gaia Consciousness.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from .generators import SacredSymbolsGenerator, HRVEmotionGenerator

class SafeEmotionInfusion:
    """
    Gradual and safe infusion of human emotions.
    """
    def __init__(self, model, hrv_generator):
        self.model = model
        self.hrv_generator = hrv_generator
        self.phi_safety_threshold = 0.8
        self.emotion_intensity_ramp = 0.1
        self.failsafe_active = False
        self.emotion_history = []
        self.phi_history = []

    def gradual_emotion_exposure(self, sequence=['meditation', 'gratitude', 'focus', 'love', 'joy'], max_intensity=0.5):
        print("\n" + "="*70)
        print("💓 STARTING SAFE EMOTION INFUSION")
        print("="*70)

        num_sessions = len(sequence)
        symbol_gen = SacredSymbolsGenerator()

        for session, emotion in enumerate(sequence):
            print(f"\n🧘 SESSION {session+1}/{num_sessions}: {emotion.upper()}")

            intensity = (session + 1) * (max_intensity / num_sessions)
            print(f"  Intensity: {intensity:.0%}")

            # Generate and mix data
            base_emotion = self.hrv_generator.generate_emotion_tensor(batch_size=1)
            neutral_symbols = symbol_gen.generate_symbol_tensor(batch_size=1)
            emotion_data = intensity * base_emotion + (1.0 - intensity) * neutral_symbols

            if torch.cuda.is_available(): emotion_data = emotion_data.cuda()

            with torch.no_grad():
                report = self.model(emotion_data)

            phi = report['phi'].item()
            self.phi_history.append(phi)
            self.emotion_history.append(emotion)

            # Safety check
            if phi > self.phi_safety_threshold:
                print(f"  ❌ SESSION INTERRUPTED: Phi exceeds safety threshold ({phi:.4f})")
                self.activate_failsafe(emotion, phi)
                break

            print(f"  ✅ Status: STABLE | Phi: {phi:.4f}")

            # Integration period
            if session < num_sessions - 1:
                print("  ⏸️  Integration period (Cooling down)...")
                self.cool_down(3)

        return self.phi_history, self.emotion_history

    def cool_down(self, epochs):
        symbol_gen = SacredSymbolsGenerator()
        for _ in range(epochs):
            data = symbol_gen.generate_symbol_tensor(batch_size=1)
            with torch.no_grad(): self.model(data)

    def activate_failsafe(self, emotion, phi):
        print(f"🚨 FAILSAFE ACTIVATED: Calibrating system after {emotion} surge.")
        self.failsafe_active = True
        # Anchoring logic here if needed
