# arkhe_companion/core/phi_core/grpc_server.py
import sys
import os
import grpc
import asyncio
from concurrent import futures

# Adicionar caminhos para imports
sys.path.append(os.path.join(os.getcwd(), 'phi_core/generated'))
sys.path.append(os.path.join(os.getcwd()))

import arkhe_pb2
import arkhe_pb2_grpc
from phi_core.phi_engine import PhiCore

class CompanionServicer(arkhe_pb2_grpc.CompanionServiceServicer):
    def __init__(self, core: PhiCore):
        self.core = core

    async def GetState(self, request, context):
        diag = self.core.get_state_diagnostic()
        emotion = arkhe_pb2.EmotionalState(
            valence=diag['emotional_state']['valence'],
            arousal=diag['emotional_state']['arousal'],
            dominance=diag['emotional_state']['dominance']
        )
        return arkhe_pb2.StateSnapshot(
            operational_phi=diag['operational_phi'],
            memory_field_energy=diag['memory_field_energy'],
            num_cognitive_spins=diag['num_cognitive_spins'],
            emotion=emotion
        )

    async def SetPhi(self, request, context):
        # Simplificado: aplica Phi diretamente por agora
        self.core.phi_operational = request.phi
        diag = self.core.get_state_diagnostic()
        return arkhe_pb2.PhiResponse(
            actual_phi=self.core.phi_operational,
            new_state=await self.GetState(None, None)
        )

    async def Connect(self, request_iterator, context):
        async for handover in request_iterator:
            if handover.type == arkhe_pb2.PERCEPTION:
                # Processar percepção
                perception = await self.core.perceive({'text': handover.payload.decode()})
                # Responder com ação (mock)
                response = await self.core.generate_response({})
                yield arkhe_pb2.Handover(
                    id="resp_" + handover.id,
                    type=arkhe_pb2.ACTION,
                    payload=response['content'].encode()
                )

async def serve():
    core = PhiCore("arkhe_grpc_node")
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    arkhe_pb2_grpc.add_CompanionServiceServicer_to_server(CompanionServicer(core), server)
    server.add_insecure_port('[::]:50051')
    print("=== Arkhe(n) Φ-Core gRPC Server starting on port 50051 ===")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())
