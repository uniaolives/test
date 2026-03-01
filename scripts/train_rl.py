import pandas as pd
import argparse
import os

def train_rl(data_path, model_name):
    # Mock de treinamento RL offline
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print("Data not found, skipping training.")
        return

    df = pd.read_parquet(data_path)
    print(f"Training on {len(df)} samples...")

    # Simula salvamento de modelo
    os.makedirs("models", exist_ok=True)
    with open(f"models/{model_name}.zip", "w") as f:
        f.write("MOCKED_MODEL_CONTENT")

    print(f"Model saved to models/{model_name}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()
    train_rl(args.data, args.model_name)
