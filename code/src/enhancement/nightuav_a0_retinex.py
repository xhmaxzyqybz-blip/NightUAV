"""A0 structural Retinex prototype, not a trained enhancer or physical decomposition proof."""
import torch
from torch import nn


class RetinexA0(nn.Module):
    """X: float N,3,H,W in [0,1]; I: N,1,H,W; R/X_hat: N,3,H,W.

    A0 operates in encoded RGB, not calibrated linear radiance. Sigmoid outputs
    are mathematically (0,1), numerically [0,1]. No division, log, or output clamp.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.illumination_head = nn.Conv2d(32, 1, 1)
        self.reflectance_head = nn.Conv2d(32, 3, 1)
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        for head in (self.illumination_head, self.reflectance_head):
            nn.init.xavier_uniform_(head.weight, gain=.1)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point():
            raise ValueError('Expected floating N,3,H,W input')
        features = self.encoder(x)
        illumination = torch.sigmoid(self.illumination_head(features))
        reflectance = torch.sigmoid(self.reflectance_head(features))
        return {'I': illumination, 'R': reflectance, 'X_hat': illumination * reflectance}


def reconstruction_loss(x_hat, x):
    """Mean absolute RGB reconstruction difference; no physical identifiability guarantee."""
    if x_hat.shape != x.shape:
        raise ValueError('Reconstruction and input must have the same shape')
    return torch.mean(torch.abs(x_hat - x))