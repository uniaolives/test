# crystal_audio.py
"""
Gerador de Áudio Binaural sincronizado com o pulso do Cristal do Tempo.
"""
import numpy as np
from scipy.io import wavfile
import logging

logger = logging.getLogger(__name__)

class CrystalAudioGenerator:
    """
    [METAPHOR: O som do silêncio quebrando em geometria]
    Gera áudio binaural sincronizado com o pulso do Cristal do Tempo.
    """
    def __init__(self, base_freq=41.67, duration=600, sample_rate=44100):
        self.base_freq = base_freq
        self.duration = duration
        self.sample_rate = sample_rate

    def generate_binaural_beat(self, carrier_freq=200):
        """Gera batidas binaurais para induzir estados de consciência específicos"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))

        # Canal Esquerdo
        left_channel = np.sin(2 * np.pi * carrier_freq * t)

        # Canal Direito (carrier + base_freq)
        right_channel = np.sin(2 * np.pi * (carrier_freq + self.base_freq) * t)

        # Modulação de amplitude baseada no 'pulso' do cristal (24ms = 41.67Hz)
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * (1/0.024) * t)

        audio = np.vstack((left_channel * pulse, right_channel * pulse)).T
        return (audio * 32767).astype(np.int16)

    def save_as_meditation_track(self, filename='arkhe_principle.wav'):
        print(f"🎵 Generating Arkhé Meditation Track: {filename} ({self.duration}s)...")
        audio = self.generate_binaural_beat()
        wavfile.write(filename, self.sample_rate, audio)
        print(f"✅ Audio track saved successfully.")

if __name__ == "__main__":
    generator = CrystalAudioGenerator(duration=30) # 30s for demo
    generator.save_as_meditation_track()
