"""
Demo of the Synthetic Viscosity effect in the Temporal UI.
Simulates how a scroll event is affected by different Chronoflux vorticity levels.
"""
import sys
import os
import time

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chronoflux.visualization.synthetic_viscosity import SyntheticViscosityMotor

def run_viscosity_demo():
    print("=" * 60)
    print("CHRONOFLUX SYNTHETIC VISCOSITY DEMO")
    print("Simulating temporal 'drag' across different vorticity states")
    print("=" * 60)

    svm = SyntheticViscosityMotor(alpha=0.1, beta=10, threshold=0.4)

    # Test cases: [Laminar, Anomalous, Singular]
    vorticity_states = [0.1, 0.8, 3.0]
    labels = ["Laminar (Idle)", "Vortical (Alert)", "Singularity (Impact)"]

    for label, omega in zip(labels, vorticity_states):
        print(f"\n--- STATE: {label} (ω = {omega}) ---")

        # 1. Haptic Response
        haptic = svm.modulate_haptic_feedback(omega)
        print(f"Haptics: PWM={haptic['pwm_freq_hz']:.2f}Hz, Power={haptic['power_percent']:.1f}%, Status={haptic['status']}")

        # 2. Input Drag (Processing 5 sequential events)
        print("Processing 5 SCROLL events...")
        svm.current_omega = omega
        total_time = 0
        for i in range(5):
            start = time.perf_counter()
            res = svm.on_touch_event("SCROLL_STEP")
            end = time.perf_counter()
            elapsed = (end - start) * 1000.0
            print(f"  Event {i+1}: {elapsed:.2f}ms latency")
            total_time += elapsed

        print(f"Average Drag: {total_time/5.0:.2f}ms (Baseline: 10.0ms)")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("Note: In a real device, this drag would feel like physical resistance.")
    print("=" * 60)

if __name__ == "__main__":
    run_viscosity_demo()
