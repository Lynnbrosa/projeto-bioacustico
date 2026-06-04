"""Loss functions for individual identification.

Three loss variants for the M3 experiment grid:

- :class:`CrossEntropyHead` — plain supervised classification, the upstream
  baseline winner.
- :class:`ArcFaceHead` — angular-margin classifier per Deng et al. 2019,
  often used in face/animal re-ID. Replicates Lapp et al.'s ArcFace baseline.
- :class:`SupConHead` — supervised contrastive loss per Khosla et al. 2020.
  No classifier weights; projects features to a 128-dim unit sphere and
  pulls same-label samples together while pushing different-label apart.

All three expose ``forward(features, labels) -> logits, loss``. SupCon's
"logits" are the cosine similarities to a running prototype per class
(useful for inference); only ``loss`` matters during training.

Torch is imported lazily so this module imports cleanly without ML extras.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch
    from torch import nn


LossName = Literal["cross_entropy", "arcface", "supcon"]


def build_loss_head(
    name: LossName,
    *,
    feature_dim: int,
    num_classes: int,
    arcface_scale: float = 30.0,
    arcface_margin: float = 0.5,
    supcon_temperature: float = 0.07,
    supcon_proj_dim: int = 128,
) -> nn.Module:
    """Factory for an embedding-to-logits + loss module."""
    if name == "cross_entropy":
        return _build_cross_entropy_head(feature_dim, num_classes)
    if name == "arcface":
        return _build_arcface_head(feature_dim, num_classes, arcface_scale, arcface_margin)
    if name == "supcon":
        return _build_supcon_head(feature_dim, num_classes, supcon_temperature, supcon_proj_dim)
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


def _build_supcon_head(
    feature_dim: int, num_classes: int, temperature: float, proj_dim: int
) -> nn.Module:
    """Supervised Contrastive (SupCon) head, Khosla et al. NeurIPS 2020.

    Projects features to a unit-sphere ``proj_dim`` space and computes the
    SupCon loss against in-batch positives (same-label samples). No class
    weights are stored; ``logits`` returned are cosine similarities to
    running per-class prototypes (mean of seen projections), useful only
    for inference. The training signal is entirely in ``loss``.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SupConHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, proj_dim),
            )
            self.register_buffer("prototypes", torch.zeros(num_classes, proj_dim))
            self.register_buffer("prototype_counts", torch.zeros(num_classes))
            self.temperature = temperature

        def forward(
            self, features: torch.Tensor, labels: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            projections = F.normalize(self.projector(features), dim=1)
            loss = _supcon_loss(projections, labels, self.temperature)

            prototypes: torch.Tensor = self.prototypes  # type: ignore[assignment]
            counts: torch.Tensor = self.prototype_counts  # type: ignore[assignment]
            with torch.no_grad():
                # Vectorised running update: scatter-add projections + counts
                # by label index. Avoids the per-class Python loop.
                prototypes.index_add_(0, labels, projections.detach())
                counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
                normalized = F.normalize(prototypes / counts.clamp(min=1).unsqueeze(1), dim=1)

            logits = projections @ normalized.t() / self.temperature
            return logits, loss

    return SupConHead()


def _supcon_loss(
    projections: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """SupCon loss for a single view per sample (in-batch positives only).

    Reference: Khosla et al. 2020, equation (2) with one augmentation per
    sample. Numerically stable via log-sum-exp.
    """
    import torch

    device = projections.device
    batch_size = projections.shape[0]
    logits = (projections @ projections.t()) / temperature
    logits_max, _ = logits.max(dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(device)
    self_mask = torch.scatter(
        torch.ones_like(mask), 1, torch.arange(batch_size, device=device).view(-1, 1), 0
    )
    mask = mask * self_mask

    exp_logits = torch.exp(logits) * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))

    positives_per_sample = mask.sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives_per_sample
    return -mean_log_prob_pos.mean()


__all__ = ["LossName", "build_loss_head"]
