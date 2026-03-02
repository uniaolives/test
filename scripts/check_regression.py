import sys

def check_regression(current_t, current_e, prod_t, prod_e):
    # Regressão se throughput cair > 20% ou erro subir > 50%
    if current_t < prod_t * 0.8:
        print(f"🚨 REGRESSION: Throughput dropped by {(1 - current_t/prod_t)*100:.1f}%")
        return True
    if current_e > prod_e * 1.5 and current_e > 0.01:
        print(f"🚨 REGRESSION: Error rate increased by {(current_e/prod_e - 1)*100:.1f}%")
        return True
    print("✅ No regression detected.")
    return False

if __name__ == "__main__":
    # Args: current_t, current_e, prod_t, prod_e
    args = [float(a) for a in sys.argv[1:]] if len(sys.argv) > 4 else [1000, 0.01, 1050, 0.01]
    if check_regression(*args):
        sys.exit(1)
    else:
        sys.exit(0)
