import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from avalon.quantum.arkhe_protocol import ArkheTherapyProtocol

def launch():
    print("🚀 LANÇANDO ARKHE-THERAPY V1.0")
    print("---------------------------------")
    protocol = ArkheTherapyProtocol(user_coherence_level=0.9) # Nível ideal
    protocol.execute_session()

if __name__ == "__main__":
    launch()
