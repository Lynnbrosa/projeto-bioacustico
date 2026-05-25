"""Loss functions for individual identification.

Two loss variants for the M3 experiment grid:

- :class:`CrossEntropyHead` — plain supervised classification, the upstream
  baseline winner.
- :class:`ArcFaceHead` — angular-margin classifier per Deng et al. 2019,
  often used in face/animal re-ID. Replicates Lapp et al.'s ArcFace baseline.

Both wrap a single ``nn.Linear`` projection from embeddings to logits and
expose a unified ``forward(features, labels) -> logits, loss`` API so the
trainer can swap between them without branching.

Torch is imported lazily so this module imports cleanly without ML extras.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch
    from torch import nn


LossName = Literal["cross_entropy", "arcface"]


def build_loss_head(
    name: LossName,
    *,
    feature_dim: int,
    num_classes: int,
    arcface_scale: float = 30.0,
    arcface_margin: float = 0.5,
) -> nn.Module:
    """Factory for an embedding-to-logits + loss module."""
    if name == "cross_entropy":
        return _build_cross_entropy_head(feature_dim, num_classes)
    if name == "arcface":
        return _build_arcface_head(feature_dim, num_classes, arcface_scale, arcface_margin)
    raise ValueError(f"unknown loss: {name}")


def _build_cross_entropy_head(feature_dim: int, num_classes: int) -> nn.Module:
    import torch.nn as nn

    class CrossEntropyHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(feature_dim, num_classes)
            self.criterion = nn.CrossEntropyLoss()

        def forward(
            self, features: torch.Tensor, labels: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits = self.fc(features)
            loss = self.criterion(logits, labels)
            return logits, loss

    return CrossEntropyHead()


def _build_arcface_head(
    feature_dim: int, num_classes: int, scale: float, margin: float
) -> nn.Module:
    """ArcFace head (Deng et al. 2019).

    Implements additive-angular-margin softmax: normalises features and class
    weights, computes cosines, adds ``margin`` to the target cosine in angle
    space, then scales by ``scale`` and feeds into cross-entropy. The margin
    is only applied during training (inferred from ``labels`` being passed).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ArcFaceHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
            nn.init.xavier_uniform_(self.weight)
            self.scale = scale
            self.margin = margin
            self.criterion = nn.CrossEntropyLoss()
            self.cos_m = math.cos(margin)
            self.sin_m = math.sin(margin)
            self.threshold = math.cos(math.pi - margin)
            self.mm = math.sin(math.pi - margin) * margin

        def forward(
            self, features: torch.Tensor, labels: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            cos = F.linear(F.normalize(features), F.normalize(self.weight))
            sin = torch.sqrt(torch.clamp(1.0 - cos.pow(2), min=0.0))
            cos_target = cos * self.cos_m - sin * self.sin_m
            cos_target = torch.where(cos > self.threshold, cos_target, cos - self.mm)
            one_hot = torch.zeros_like(cos)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            output = one_hot * cos_target + (1.0 - one_hot) * cos
            logits = output * self.scale
            loss = self.criterion(logits, labels)
            return logits, loss

    return ArcFaceHead()


__all__ = ["LossName", "build_loss_head"]
