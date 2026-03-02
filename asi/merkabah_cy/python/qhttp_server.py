# asi/merkabah_cy/python/qhttp_server.py
import grpc
from concurrent import futures
import sys
import os

# Add proto dir to path to find generated files
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'proto'))

try:
    import qhttp_pb2
    import qhttp_pb2_grpc
except ImportError:
    # Fallback for when stubs are not yet generated
    print("Warning: qhttp stubs not found. Run protoc to generate them.")
    class Dummy: pass
    qhttp_pb2 = Dummy()
    qhttp_pb2_grpc = Dummy()
    qhttp_pb2_grpc.QHTTPServicer = object

from qiskit import QuantumCircuit, Aer, execute
import json

class QHTTPServicer(qhttp_pb2_grpc.QHTTPServicer):
    def __init__(self):
        self.backend = Aer.get_backend('statevector_simulator')

    def GetState(self, request, context):
        n = 4
        qc = QuantumCircuit(n)
        qc.h(range(n))
        job = execute(qc, self.backend)
        state = job.result().get_statevector()
        return qhttp_pb2.QuantumState(
            real=state.real.tolist(),
            imag=state.imag.tolist(),
            n_qubits=n,
            basis="computational"
        )

    def Evolve(self, request, context):
        state_data = request.state
        amplitudes = [complex(r, i) for r, i in zip(state_data.real, state_data.imag)]
        n = state_data.n_qubits
        qc = QuantumCircuit(n)
        qc.initialize(amplitudes, range(n))
        qc.h(range(n))
        job = execute(qc, self.backend)
        final = job.result().get_statevector()
        return qhttp_pb2.EvolveResponse(
            final_state=qhttp_pb2.QuantumState(
                real=final.real.tolist(),
                imag=final.imag.tolist(),
                n_qubits=n,
                basis=state_data.basis
            ),
            fidelity=0.99
        )

    def Entangle(self, request, context):
        return qhttp_pb2.EntangleResponse(
            entanglement_id="epr_" + request.target_module,
            bell_state=qhttp_pb2.QuantumState(
                real=[0.70710678, 0.0, 0.0, 0.70710678],
                imag=[0.0, 0.0, 0.0, 0.0],
                n_qubits=2,
                basis="bell"
            )
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    qhttp_pb2_grpc.add_QHTTPServicer_to_server(QHTTPServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("Servidor qhttp rodando na porta 50051...")
    # server.start()
    # server.wait_for_termination()

if __name__ == '__main__':
    serve()
