import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from avalon.quantum.crystal_audio import CrystalAudioGenerator

def generate_soundtrack():
    print("🔊 Gerando trilha sonora do Arkhé...")
    # 10 minutos = 600 segundos
    generator = CrystalAudioGenerator(duration=600)
    generator.save_as_meditation_track('arkhe_principle.wav')

if __name__ == "__main__":
    generate_soundtrack()
