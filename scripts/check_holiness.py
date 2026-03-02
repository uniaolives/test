# scripts/check_holiness.py
import argparse
import hashlib

def get_holiness_score(author):
    # Deterministic score based on email for simulation
    h = hashlib.sha256(author.encode()).hexdigest()
    score = int(h[:2], 16) / 25.6 # 0 to 10 scale
    return score + 5.0 # Base holiness

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", required=True)
    args = parser.parse_args()

    score = get_holiness_score(args.author)
    print(f"{score:.2f}")
