import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

class GLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, meta_neurons=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, meta_neurons)
        )
    def forward(self, x):
        return self.encoder(x)

model = GLP()
# In a real setup, we would load the weights. For simulation:
# model.load_state_dict(torch.load('/model/model.pt', map_location='cpu'))
model.eval()

@app.route('/encode', methods=['POST'])
def encode():
    data = request.json
    x = torch.tensor(data['activation']).float()
    with torch.no_grad():
        meta = model(x).tolist()
    return jsonify({'meta': meta})

@app.route('/steer', methods=['POST'])
def steer():
    data = request.json
    meta = np.array(data['meta'])
    direction = np.array(data['direction'])
    strength = data.get('strength', 1.0)
    steered = meta + strength * direction
    return jsonify({'steered': steered.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
