#!/usr/bin/env python3
"""
Arkhe(n) Diplomatic Simulator (Resilient Edition)
Bridges GNU Radio handshakes to the diplomatic logic and manages state transitions.
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
    GOLDEN_ALPHA = 0.618
    SEMION_ALPHA = 0.5

    state = "NORMAL"
    current_alpha = GOLDEN_ALPHA

    while True:
        try:
            message = socket.recv_json()
            # print(f"📥 Received {message['type']} from {message['node_id']}")

            if message['type'] == 'HANDSHAKE_REQUEST':
                coherence_local = message['coherence_local']
                # Simulate external interference or chaos
                # In a real HIL, this comes from the SDR/Atmosphere
                coherence_remote = message.get('remote_coherence_sim', 0.98)

                avg_coherence = (coherence_local + coherence_remote) / 2.0

                status = "ACCEPTED"

                if avg_coherence < PSI:
                    if state != "SEMIONIC":
                        print(f"⚠️ [SIM] Coherence dropped to {avg_coherence:.4f}. Switching to SEMIONIC FALLBACK.")
                        state = "SEMIONIC"
                        current_alpha = SEMION_ALPHA
                        status = "SemionicFallback"
                    else:
                        status = "REJECTED"
                elif state == "SEMIONIC":
                    print(f"💡 [SIM] Coherence recovered to {avg_coherence:.4f}. Starting Annealing.")
                    state = "ANNEALING"

                if state == "ANNEALING":
                    # Step annealing
                    current_alpha += 0.01
                    if current_alpha >= GOLDEN_ALPHA:
                        current_alpha = GOLDEN_ALPHA
                        state = "NORMAL"
                        print("✅ [SIM] Annealing complete. System back to NORMAL.")

                # g|ψ|² = -Δϕ
                g_adjustment = -message['phase_remote']

                response = {
                    "status": status,
                    "g_adjustment": g_adjustment,
                    "coherence_global": avg_coherence,
                    "alpha": current_alpha,
                    "protocol_state": state
                }

                # if status != "ACCEPTED":
                #    print(f"📤 Response: {status} (Coherence: {avg_coherence:.4f}, Alpha: {current_alpha:.4f})")

                socket.send_json(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            socket.send_json({"status": "ERROR", "message": str(e)})

if __name__ == "__main__":
    main()
