"""Supervised training loop for the feature extractor.

Single ``train_supervised`` entry point parameterised by backbone and loss so
the M3 grid (ResNet18/50/EfficientNet/ConvNeXt by CrossEntropy/ArcFace) can be
swept by calling it with different configs.

Heavy imports (torch, opensoundscape) are kept lazy so the module imports
cleanly without ML extras installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from bioacid.losses import LossName
from bioacid.models import BackboneName

if TYPE_CHECKING:
    import torch
    from torch import nn


def _identity(x: Any) -> Any:
    return x


@dataclass
class TrainingHistory:
    """Per-epoch losses and accuracies recorded during training."""

    epoch_loss: list[float] = field(default_factory=list)
    epoch_top1: list[float] = field(default_factory=list)


@dataclass
class TrainConfig:
    """Knobs for :func:`train_supervised`."""

    backbone: BackboneName = "resnet18"
    loss: LossName = "cross_entropy"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 42
    arcface_scale: float = 30.0
    arcface_margin: float = 0.5


def train_supervised(
    train_df: pd.DataFrame,
    *,
    preprocessor: Any,
    num_classes: int,
    label_to_index: dict[int, int],
    config: TrainConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, nn.Module, TrainingHistory]:
    """Train a backbone + loss-head end-to-end on ``train_df``.

    Returns ``(backbone, loss_head, history)``. The backbone is the embedding
    network alone; the loss head holds the classifier weights. To embed at
    inference time, just call the backbone.
    """
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from bioacid.data import AIIDLocalizedClipDataset
    from bioacid.losses import build_loss_head
    from bioacid.models import backbone_feature_dim, build_backbone

    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)

    dataset = AIIDLocalizedClipDataset(
        train_df, preprocessor=preprocessor, bypass_augmentations=False
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_identity,
    )

    backbone = build_backbone(cfg.backbone, pretrained=False).to(device)
    feature_dim = backbone_feature_dim(cfg.backbone)
    loss_head = build_loss_head(
        cfg.loss,
        feature_dim=feature_dim,
        num_classes=num_classes,
        arcface_scale=cfg.arcface_scale,
        arcface_margin=cfg.arcface_margin,
    ).to(device)

    params = list(backbone.parameters()) + list(loss_head.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = TrainingHistory()

    for epoch in range(cfg.epochs):
        backbone.train()
        loss_head.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False):
            x = torch.vstack([s.data[None, :, :] for s in batch]).to(device)
            y = torch.tensor(
                [label_to_index[int(s.aiid_label)] for s in batch],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            features = backbone(x)
            logits, loss = loss_head(features, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch)
            running_correct += int((logits.argmax(dim=1) == y).sum().item())
            running_total += len(batch)

        epoch_loss = running_loss / running_total
        epoch_top1 = running_correct / running_total
        history.epoch_loss.append(epoch_loss)
        history.epoch_top1.append(epoch_top1)
        print(f"epoch {epoch + 1:>3}/{cfg.epochs}  loss={epoch_loss:.4f}  top1={epoch_top1:.3f}")

    return backbone, loss_head, history


__all__ = ["TrainConfig", "TrainingHistory", "train_supervised"]
