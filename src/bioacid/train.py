"""Supervised classification training loop for the feature extractor.

Single-call ``train_supervised`` that walks a labeled-clip dataframe through
the preprocessor and a ResNet18 classifier with cross-entropy loss. Returns
the trained model along with a small history dict (epoch losses) for logs.

This is the minimum viable training entry point for M2: enough to validate
the end-to-end pipeline (data → preprocess → backbone → loss → optim) using
the ported modules. Heavier features (samplers, ArcFace, SpecCon, scheduling,
W&B integration) come in M3.

Torch + opensoundscape are imported lazily so this module imports cleanly in
a dev-only environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

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

    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 42


def train_supervised(
    train_df: pd.DataFrame,
    *,
    preprocessor: Any,
    num_classes: int,
    label_to_index: dict[int, int],
    config: TrainConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, TrainingHistory]:
    """Train a ``Resnet18Classifier`` with supervised cross-entropy.

    The ``train_df`` is expected to follow the upstream sample CSV format
    (``file``, ``song_center_time``, ``aiid_label`` columns). ``label_to_index``
    maps original integer labels to ``[0, num_classes)`` indices required by
    ``CrossEntropyLoss``.

    Returns the trained ``nn.Module`` and a :class:`TrainingHistory` with the
    average loss and top-1 accuracy per epoch.
    """
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from bioacid.data import AIIDLocalizedClipDataset
    from bioacid.models import Resnet18Classifier

    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)

    dataset = AIIDLocalizedClipDataset(
        train_df,
        preprocessor=preprocessor,
        bypass_augmentations=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_identity,
    )

    model = Resnet18Classifier(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = TrainingHistory()

    for epoch in range(cfg.epochs):
        model.train()
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
            _, logits = model(x)
            loss = loss_fn(logits, y)
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

    return model, history


__all__ = ["TrainConfig", "TrainingHistory", "train_supervised"]
