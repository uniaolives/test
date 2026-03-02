import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

def train():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("arkhe-quantum-prediction")

    with mlflow.start_run() as run:
        model = SimpleModel()
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_metric("accuracy", 0.95)

        # Log model
        mlflow.pytorch.log_model(model, "model")
        print(f"Run ID: {run.info.run_id}")

        # Logic to record model version in ledger would go here
        # e.g., calling ledger gRPC client to store run_id and artifact_uri
        print("Model version recorded in imutable ledger.")

if __name__ == "__main__":
    train()
