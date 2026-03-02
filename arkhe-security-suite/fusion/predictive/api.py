from flask import Flask, jsonify, request
import torch
import numpy as np
from lstm_model import EntropyLSTM

app = Flask(__name__)
model = EntropyLSTM()

@app.route('/predict/<node_id>', methods=['GET'])
def predict(node_id):
    # Simulated inference
    dummy_input = torch.randn(1, 50, 1)
    with torch.no_grad():
        pred = model(dummy_input).numpy().flatten()

    failure_prob = float(np.mean(pred > 0.8))
    return jsonify({
        "node_id": node_id,
        "failure_probability": failure_prob,
        "predictions": pred.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
