"""Backbone architectures for the feature extractor.

Torch / torchvision / opensoundscape imports are kept inside function bodies
so the module can be imported in a dev-only environment (no ML extras
installed) without crashing — useful for CI smoke tests.

For M2 we expose:

- :func:`build_resnet18_1ch`: ResNet18 stub with single-channel input and no
  classifier head, suitable as a feature extractor.
- :class:`Resnet18Classifier`: full classifier (embedder + linear head) that
  mirrors upstream ``Resnet18_Classifier`` so the released Ovenbird checkpoint
  state-dict loads cleanly.
- :func:`load_ovenbird_checkpoint`: build the architecture, load the
  state-dict and wrap in an :class:`opensoundscape.CNN` ready for ``embed``.

Ported from ``external/upstream/src/model.py`` (Lapp et al. 2025).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from opensoundscape import CNN
    from torch import nn


NUM_CLASSES_PAPER = 234
"""Number of individual Ovenbirds in the released checkpoint's training set."""


def build_resnet18_1ch() -> nn.Module:
    """Construct a ResNet18 with 1-channel conv1 and identity FC head.

    No pretrained ImageNet weights are loaded — callers either train from
    scratch or replace the weights via ``load_state_dict``.
    """
    import torch.nn as nn
    import torchvision.models as tvm
    from opensoundscape.ml import cnn_architectures

    model = tvm.resnet18(weights=None)
    model.conv1 = cnn_architectures.change_conv2d_channels(model.conv1, num_channels=1)
    model.fc = nn.Identity()
    model.constructor_name = "resnet18_1ch_embedder"
    return model  # type: ignore[no-any-return]


def _build_classifier(num_classes: int) -> nn.Module:
    import torch.nn as nn

    class Resnet18Classifier(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.embedder = build_resnet18_1ch()
            self.classifier = nn.Linear(512, num_classes)
            self.constructor_name = "Resnet18_Classifier"

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            emb = self.embedder(x)
            logits = self.classifier(emb)
            return emb, logits

    return Resnet18Classifier(num_classes=num_classes)


def Resnet18Classifier(num_classes: int) -> nn.Module:
    """Factory for the Ovenbird-style classifier (embedder + linear head)."""
    return _build_classifier(num_classes=num_classes)


def load_ovenbird_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    sample_duration: float = 2.0,
    num_classes: int = NUM_CLASSES_PAPER,
) -> CNN:
    """Build the architecture, load the checkpoint and return a ready CNN.

    The returned ``opensoundscape.CNN`` has ``embedding_layer="avgpool"`` so a
    subsequent call to ``cnn.embed(df)`` yields 512-dim feature vectors —
    matching the upstream demo behaviour.
    """
    import torch
    from opensoundscape import CNN

    from bioacid.preprocessor import OvenbirdPreprocessor

    classifier = _build_classifier(num_classes=num_classes)
    state_dict = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    cnn: Any = CNN(classifier.embedder, sample_duration=sample_duration, classes=list(range(512)))
    preproc = OvenbirdPreprocessor()
    preproc.pipeline.load_audio.set(load_metadata=False)
    cnn.preprocessor = preproc
    cnn.network.embedding_layer = "avgpool"
    return cnn


__all__ = [
    "NUM_CLASSES_PAPER",
    "Resnet18Classifier",
    "build_resnet18_1ch",
    "load_ovenbird_checkpoint",
]
