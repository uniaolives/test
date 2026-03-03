import torch
import torch.nn as nn

class EntropyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, horizon=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
