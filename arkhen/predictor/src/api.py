from flask import Flask, request, jsonify
import torch
import numpy as np
from model import EntropyLSTM

app = Flask(__name__)
model = EntropyLSTM()

@app.route('/predict/<node_id>', methods=['GET'])
def predict(node_id):
    # Mock recent series of 50 samples
    recent_series = np.random.rand(50)
    input_tensor = torch.FloatTensor(recent_series).unsqueeze(0).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).numpy().flatten()

    prob = float(np.mean(pred > 0.8))
    return jsonify({"node_id": node_id, "failure_probability": prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
