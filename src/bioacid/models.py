"""Backbone architectures and classifier heads.

Provides a pluggable backbone factory (``build_backbone``) covering the
M3 grid (ResNet18, ResNet50, EfficientNet-B0, ConvNeXt-Tiny via ``timm``)
and a classifier head that mirrors the upstream Ovenbird checkpoint format
so the released state-dict can be loaded directly.

Torch / torchvision / timm / opensoundscape are imported lazily so the
module can be imported in a dev-only environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch
    from opensoundscape import CNN
    from torch import nn


NUM_CLASSES_PAPER = 234
"""Number of individual Ovenbirds in the released checkpoint's training set."""


BackboneName = Literal["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"]
"""Supported backbones for the M3 experiment grid."""


_FEATURE_DIM: dict[str, int] = {
    "resnet18": 512,
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "convnext_tiny": 768,
}


def backbone_feature_dim(name: BackboneName) -> int:
    """Output feature dimension of the named backbone after pooling."""
    if name not in _FEATURE_DIM:
        raise ValueError(f"unknown backbone: {name}")
    return _FEATURE_DIM[name]


def build_backbone(name: BackboneName, *, pretrained: bool = False) -> nn.Module:
    """Build a 1-channel feature extractor with no classifier head.

    Uses ``torchvision`` for ResNet (so the upstream Ovenbird checkpoint
    state-dict loads cleanly) and ``timm`` for the rest. ``pretrained=False``
    means no weights are downloaded — callers either train from scratch or
    replace via ``load_state_dict``. ``pretrained=True`` requires network
    access to the relevant model hub.
    """
    import torch.nn as nn
    import torchvision.models as tvm
    from opensoundscape.ml import cnn_architectures

    if name == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.conv1 = cnn_architectures.change_conv2d_channels(model.conv1, num_channels=1)
        model.fc = nn.Identity()
    elif name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.conv1 = cnn_architectures.change_conv2d_channels(model.conv1, num_channels=1)
        model.fc = nn.Identity()
    elif name in ("efficientnet_b0", "convnext_tiny"):
        import timm

        model = timm.create_model(name, pretrained=pretrained, in_chans=1, num_classes=0)
    else:
        raise ValueError(f"unknown backbone: {name}")

    model.constructor_name = f"{name}_1ch_embedder"
    return model  # type: ignore[no-any-return]


def build_resnet18_1ch() -> nn.Module:
    """Compatibility alias for the M1/M2 callers."""
    return build_backbone("resnet18", pretrained=False)


def _build_classifier(num_classes: int, backbone: BackboneName = "resnet18") -> nn.Module:
    """Build a ``backbone + linear head`` classifier."""
    import torch.nn as nn

    feature_dim = backbone_feature_dim(backbone)

    class Classifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedder = build_backbone(backbone, pretrained=False)
            self.classifier = nn.Linear(feature_dim, num_classes)
            self.constructor_name = f"{backbone}_classifier"

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            emb = self.embedder(x)
            logits = self.classifier(emb)
            return emb, logits

    return Classifier()


def Resnet18Classifier(num_classes: int) -> nn.Module:
    """Factory for the Ovenbird-style ResNet18 classifier (M1/M2 alias)."""
    return _build_classifier(num_classes=num_classes, backbone="resnet18")


def build_classifier(num_classes: int, backbone: BackboneName = "resnet18") -> nn.Module:
    """Build a classifier for any supported backbone."""
    return _build_classifier(num_classes=num_classes, backbone=backbone)


def load_ovenbird_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    sample_duration: float = 2.0,
    num_classes: int = NUM_CLASSES_PAPER,
) -> CNN:
    """Build the ResNet18 classifier, load the upstream checkpoint and return a ready CNN."""
    import torch
    from opensoundscape import CNN

    from bioacid.preprocessor import OvenbirdPreprocessor

    classifier = _build_classifier(num_classes=num_classes, backbone="resnet18")
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
    "BackboneName",
    "Resnet18Classifier",
    "backbone_feature_dim",
    "build_backbone",
    "build_classifier",
    "build_resnet18_1ch",
    "load_ovenbird_checkpoint",
]
