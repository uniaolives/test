import argparse
import sys

def validate_model(model_path, threshold):
    # Mock de validação
    print(f"Validating model at {model_path} against threshold {threshold}...")
    # Simula F1-score
    f1 = 0.85
    if f1 < threshold:
        print(f"Validation FAILED (F1={f1})")
        sys.exit(1)
    else:
        print(f"Validation PASSED (F1={f1})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    validate_model(args.model, args.threshold)
