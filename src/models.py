from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CompactEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x).flatten(1))


class Classifier(nn.Module):
    def __init__(self, encoder: CompactEncoder, feature_dim: int = 128, classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class ProjectionHead(nn.Module):
    def __init__(self, feature_dim: int = 128, projection_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(),
                                 nn.Linear(feature_dim, projection_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, encoder: CompactEncoder, feature_dim: int = 128, projection_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(feature_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.encoder(x)), dim=1)


def make_linear_probe(encoder: CompactEncoder, feature_dim: int = 128) -> Classifier:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    return Classifier(encoder, feature_dim)

