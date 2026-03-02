import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from avalon.biological.eeg_processor import RealEEGProcessor
from avalon.quantum.arkhe_protocol import ArkheTherapyProtocol

def test_integration():
    print("🧪 Testando integração básica Arkhé + Biofeedback...")

    # 1. Simular coleta de EEG
    eeg = RealEEGProcessor(device_type='synthetic')
    eeg.connect()
    eeg.start_stream()
    coherence = eeg.get_coherence()
    print(f"📊 Coerência inicial detectada: {coherence:.4f}")

    # 2. Iniciar Protocolo Terapêutico baseado na coerência
    print("🧘 Iniciando protocolo terapêutico...")
    protocol = ArkheTherapyProtocol(user_coherence_level=coherence)
    result = protocol.execute_session()

    print(f"🏁 Resultado do teste: {result}")
    eeg.stop()

if __name__ == "__main__":
    test_integration()
