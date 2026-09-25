from __future__ import annotations

import torch
from torch.nn import functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Symmetric NT-Xent over 2N views; each other view is its sole positive."""
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("z1 and z2 must have identical [batch, dimension] shapes")
    if z1.shape[0] < 2 or temperature <= 0:
        raise ValueError("batch size must be >=2 and temperature >0")
    batch = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    logits = z @ z.T / temperature
    diagonal = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    positives = (torch.arange(2 * batch, device=z.device) + batch) % (2 * batch)
    return F.cross_entropy(logits, positives)
