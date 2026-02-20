#!/usr/bin/env python3
"""
Arkhe(n) Diplomatic Simulator
Bridges GNU Radio handshakes to the diplomatic logic.
"""

import zmq
import json
import time

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5556")

    print("📡 [DIPLOMATIC SIM] Listening for handshakes on port 5556...")

    # Threshold Ψ
    PSI = 0.847

    while True:
        try:
            message = socket.recv_json()
            print(f"📥 Received {message['type']} from {message['node_id']}")

            if message['type'] == 'HANDSHAKE_REQUEST':
                coherence_local = message['coherence_local']
                coherence_remote = 0.98 # Simulated remote coherence

                avg_coherence = (coherence_local + coherence_remote) / 2.0

                if avg_coherence >= PSI:
                    status = "ACCEPTED"
                    # g|ψ|² = -Δϕ
                    # Simplified adjustment
                    g_adjustment = -message['phase_remote']
                else:
                    status = "REJECTED"
                    g_adjustment = 0.0

                response = {
                    "status": status,
                    "g_adjustment": g_adjustment,
                    "coherence_global": avg_coherence
                }

                print(f"📤 Response: {status} (Coherence: {avg_coherence:.4f})")
                socket.send_json(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            socket.send_json({"status": "ERROR", "message": str(e)})

if __name__ == "__main__":
    main()
