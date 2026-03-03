# arkhe_omni_system/applied_ecosystems/rf_entropy_node/rf_seed_injector.py
# Conecta o extrator de entropia RF ao nó Arkhe(N) via gRPC

import time
import grpc
import sys
import os

# Add network protocol to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../network_mesh/protocol')))

try:
    import arkhe_network_pb2
    import arkhe_network_pb2_grpc
except ImportError:
    # Fallback to local import if path not set
    try:
        from arkhe_omni_system.arkhe_qutip import arkhe_network_pb2
        from arkhe_omni_system.arkhe_qutip import arkhe_network_pb2_grpc
    except ImportError:
        # Mock for development
        class MockResponse:
            accepted = True
            phi_target = 0.85
        class MockStub:
            def SubmitAtmosphericSeed(self, msg): return MockResponse()
        arkhe_network_pb2 = type('Mock', (), {'AtmosphericSeed': lambda **k: k})
        arkhe_network_pb2_grpc = type('Mock', (), {'ArkheHypergraphStub': lambda c: MockStub()})

class ArkheRFSeedInjector:
    """
    Conecta o extrator de entropia RF ao nó Arkhe(N).
    """

    def __init__(self, node_id, grpc_endpoint):
        self.node_id = node_id
        self.grpc_endpoint = grpc_endpoint

    def on_seed_generated(self, seed_bytes, phi_rf):
        """
        Callback quando nova seed atmosférica é gerada.
        Envia para o nó Arkhe(N) via gRPC.
        """
        try:
            channel = grpc.insecure_channel(self.grpc_endpoint)
            stub = arkhe_network_pb2_grpc.ArkheHypergraphStub(channel)

            # Construir mensagem de seed atmosférica
            seed_msg = arkhe_network_pb2.AtmosphericSeed(
                node_id=self.node_id,
                seed_hash=seed_bytes,
                phi_atmospheric=phi_rf,
                source_device='RTL-SDR-V4',
                frequency_mhz=5.0,
                timestamp_unix=int(time.time())
            )

            # Enviar para a testnet
            response = stub.SubmitAtmosphericSeed(seed_msg)

            if response.accepted:
                print(f"✅ Seed atmosférica aceita! Φ_RF = {phi_rf:.4f}")
                return True
            else:
                print(f"❌ Seed rejeitada.")
                return False
        except Exception as e:
            print(f"⚠️ Error injecting seed: {e}")
            return False

if __name__ == "__main__":
    injector = ArkheRFSeedInjector("ARKHE-RF-001", "localhost:50051")
    injector.on_seed_generated(b"dummy_seed_32_bytes_long_entropy", 0.75)
