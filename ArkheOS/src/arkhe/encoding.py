import torch
import torch.nn as nn

class GridEncoder(nn.Module):
    """
    Rich grid encoding for ARC tasks.
    Preserves 2D structure using a CNN.
    """
    def __init__(self, output_dim=384):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Flatten(),
            nn.Linear(32*64, output_dim)  # 8*8*32 = 2048 -> 384
        )
    def forward(self, x):
        # x: (batch, 1, H, W) with values 0-9
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return self.conv(x.float())
