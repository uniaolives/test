import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RecursiveRankTensor(nn.Module):
    """
    A tensor module that can dynamically expand its hidden dimension
    based on the entropy (surprise) of the input.
    """
    def __init__(self, initial_dim=64, growth_factor=1.5, entropy_threshold=0.7):
        super().__init__()
        self.current_dim = initial_dim
        self.growth_factor = growth_factor
        self.entropy_threshold = entropy_threshold
        self.projection = nn.Linear(initial_dim, initial_dim)

    def forward(self, x):
        # Ensure x is at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Calculate entropy as a measure of 'surprise'
        with torch.no_grad():
            # Calculate entropy on the existing feature dimension
            probs = F.softmax(x, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()

        if entropy > self.entropy_threshold:
            self._expand_rank()

        # Realignment/Padding of input if dimensions don't match
        if x.size(-1) < self.current_dim:
            padding = torch.zeros(*x.shape[:-1], self.current_dim - x.size(-1)).to(x.device)
            x = torch.cat([x, padding], dim=-1)
        elif x.size(-1) > self.current_dim:
            # This shouldn't happen with internal growth but handles external mismatches
            self._realign_dimension(x.size(-1))

        return self.projection(x)

    def _realign_dimension(self, new_dim):
        # Handle mismatch during external rank expansion
        self.projection = nn.Linear(new_dim, new_dim).to(self.projection.weight.device)
        self.current_dim = new_dim

    def _expand_rank(self):
        new_dim = int(self.current_dim * self.growth_factor)

        # Create new projection layer with expanded dimensions
        old_weight = self.projection.weight.data
        old_bias = self.projection.bias.data

        new_projection = nn.Linear(new_dim, new_dim).to(old_weight.device)

        # Initialize new projection with old weights (identity padding)
        with torch.no_grad():
            new_projection.weight.fill_(0.0)
            d_out = min(old_weight.size(0), new_dim)
            d_in = min(old_weight.size(1), new_dim)
            new_projection.weight[:d_out, :d_in] = old_weight[:d_out, :d_in]
            new_projection.bias.fill_(0.0)
            new_projection.bias[:d_out] = old_bias[:d_out]

        self.projection = new_projection
        self.current_dim = new_dim

class AutoreferentialLoss(nn.Module):
    """
    Loss function that penalizes lack of growth and rewards
    structural evolution (measured by rank expansion).
    """
    def __init__(self, base_loss_fn=nn.MSELoss(), growth_weight=0.1):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.growth_weight = growth_weight

    def forward(self, pred, target, model_rank):
        # Ensure pred and target match for base loss
        if pred.shape != target.shape:
            # Pad target if needed (simplified for demo)
            if pred.size(-1) > target.size(-1):
                padding = torch.zeros(*target.shape[:-1], pred.size(-1) - target.size(-1)).to(target.device)
                target = torch.cat([target, padding], dim=-1)
            else:
                target = target[..., :pred.size(-1)]

        base_loss = self.base_loss_fn(pred, target)

        # Growth reward: lower loss for higher rank (incentivizes expansion)
        growth_reward = self.growth_weight * (1.0 / (1.0 + np.log(float(model_rank))))

        return base_loss + growth_reward
