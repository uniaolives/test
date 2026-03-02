import grpc
import time
from concurrent import futures
import sys
import os

try:
    from . import arkhe_network_pb2
    from . import arkhe_network_pb2_grpc
except ImportError:
    import arkhe_network_pb2
    import arkhe_network_pb2_grpc

# ARKHE(N) HYPERGRAPH NODE SERVER
# "Conectando nós globais via ressonância gRPC."

class ArkheHypergraphServicer(arkhe_network_pb2_grpc.ArkheHypergraphServicer):
    """
    Implementation of the Arkhe Hypergraph consensus and telemetry service.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.phi_target = 0.847
        self.ledger = []
        self.active_entanglements = 0

    def SubmitHandover(self, request, context):
        """
        Processes a Proof-of-Coherence mining proposal from a peer.
        """
        print(f"📡 [{self.node_id}] Handover recebido de {request.node_id}")

        # 1. Validação do Proof-of-Coherence (Φ threshold)
        if request.phi_achieved >= self.phi_target:
            print(f"   ✅ [VALID] Handover aceito! Φ = {request.phi_achieved:.4f}")
            # Registrar no ledger Ω+∞
            self.ledger.append(request.block_hash)
            self.active_entanglements += 1

            return arkhe_network_pb2.HandoverResponse(
                accepted=True,
                message="Ressonância confirmada. Bloco adicionado ao Ledger.",
                network_phi_target=self.phi_target + 0.0001 # Aumenta dificuldade marginalmente
            )
        else:
            print(f"   ❌ [REJECTED] Coerência insuficiente: {request.phi_achieved:.4f}")
            return arkhe_network_pb2.HandoverResponse(
                accepted=False,
                message="Decoerência detectada. Proposta inválida.",
                network_phi_target=self.phi_target
            )

    def QckdExchange(self, request, context):
        """
        Simulates the Quantum Coherence Key Distribution handshake.
        Matching bases establishes a shared context.
        """
        print(f"🔐 [{self.node_id}] QCKD handshake com {request.node_id}")
        # In a real implementation, we would compare the basis_sequence
        # and return indices of matches.
        return arkhe_network_pb2.QckdBasisResponse(
            matching_bases=b"\x01\x03\x07", # Mocked matches
            bell_violation_score=2.82 # Violating Bell's inequality (S > 2)
        )

    def GetNodeCoherence(self, request, context):
        """
        Returns real-time telemetry from the node.
        """
        mode = "TRANSCEND" if self.phi_target > 0.9 else "STABILIZE"
        return arkhe_network_pb2.CoherenceMetrics(
            current_phi=0.912, # Mocked current node Φ
            active_entanglements=self.active_entanglements,
            operational_mode=mode
        )

def serve_arkhe_node(node_id="ARKHE_GENESIS", port=50051):
    """
    Starts the gRPC server for an Arkhe(N) node.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    arkhe_network_pb2_grpc.add_ArkheHypergraphServicer_to_server(
        ArkheHypergraphServicer(node_id), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"🌍 Nó Arkhe(N) [{node_id}] Online na porta {port}.")
    print("   Aguardando conexões quânticas (gRPC)...")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    # Default execution for local testing
    serve_arkhe_node()
