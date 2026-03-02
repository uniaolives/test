import sys
import os

def check_quality(throughput, error_rate):
    # Mock limits
    throughput_min = 500
    error_max = 0.05

    if throughput >= throughput_min and error_rate <= error_max:
        print(f"✅ Quality OK: throughput={throughput}, error_rate={error_rate}")
        return True
    else:
        print(f"❌ Quality insufficient: throughput={throughput} (min {throughput_min}), error_rate={error_rate} (max {error_max})")
        return False

if __name__ == "__main__":
    # Simulação de argumentos: throughput, error_rate
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
    e = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

    if check_quality(t, e):
        sys.exit(0)
    else:
        sys.exit(1)
