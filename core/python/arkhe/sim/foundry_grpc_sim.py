import grpc
import arkhe_pb2
import arkhe_pb2_grpc
import json
import time
import random
from datetime import datetime

def run_grpc_sim():
    print("🚀 Arkhe-Foundry Advanced gRPC Simulator (Ω+211)")

    channel = grpc.insecure_channel('localhost:50051')
    stub = arkhe_pb2_grpc.ArkheServiceStub(channel)

    object_types = ["SupplyChainAlert", "Vessel", "Flight", "SmartContract"]

    try:
        while True:
            obj_type = random.choice(object_types)
            obj_id = f"obj-{random.randint(100, 999)}"
            phi = 0.618 + random.uniform(-0.1, 0.1)

            payload = {
                "phi": phi,
                "entropy": 0.618,
                "status": "operational",
                "last_sync": datetime.now().isoformat()
            }

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Syncing Foundry Object: {obj_id} ({obj_type})")

            request = arkhe_pb2.OntologyRequest(
                object_id=obj_id,
                object_type=obj_type,
                payload_json=json.dumps(payload)
            )

            try:
                response = stub.UpdateOntology(request)
                print(f"  gRPC Response: success={response.success}, msg='{response.message}'")
            except grpc.RpcError as e:
                print(f"  gRPC Error: {e.code()} - {e.details()}")

            time.sleep(random.uniform(3, 7))

    except KeyboardInterrupt:
        print("\nStopping simulator...")

if __name__ == "__main__":
    run_grpc_sim()
