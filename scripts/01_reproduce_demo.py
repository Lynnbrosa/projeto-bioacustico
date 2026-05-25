"""M1: Reproduce the upstream demo on the public Ovenbird sample dataset.

Loads the pretrained Ovenbird feature extractor checkpoint, generates
embeddings for the 100 sample clips (10 individuals), and saves them as a
parquet artifact under ``data/processed/sample_embeddings.parquet``.

Why we don't call upstream's ``load_ovenbird_model`` directly: that function
instantiates ``torchvision.models.resnet18(weights=IMAGENET1K_V1)`` which
triggers a download of ImageNet pretrained weights from
``download.pytorch.org`` — blocked by the sandbox network policy. Since the
Ovenbird checkpoint state-dict overwrites every parameter anyway, we build
the same architecture with ``weights=None`` and load the checkpoint directly.

Usage:
    uv run --extra ml --extra dev scripts/01_reproduce_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from opensoundscape import CNN
from opensoundscape.ml import cnn_architectures
from sklearn.metrics.pairwise import cosine_distances

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "upstream"

if not UPSTREAM.exists():
    print(
        f"Upstream repo not found at {UPSTREAM}. Clone it with:\n"
        "  git clone https://github.com/sammlapp/ovenbird-individual-recognition.git "
        f"{UPSTREAM}",
        file=sys.stderr,
    )
    raise SystemExit(1)

sys.path.insert(0, str(UPSTREAM / "src"))

from preprocessor import OvenbirdPreprocessor  # type: ignore[import-not-found]  # noqa: E402

CHECKPOINT = UPSTREAM / "checkpoints" / "full_2025-04-10T11:02:36.028451_best.pth"
CSV_PATH = UPSTREAM / "sample_data" / "labeled_clips_sample.csv"
OUTPUT = ROOT / "data" / "processed" / "sample_embeddings.parquet"
NUM_CLASSES_PAPER = 234  # training had 234 individuals across the full PAM dataset


def build_resnet18_1ch() -> nn.Module:
    """ResNet18 with single-channel input, no classifier head, no ImageNet download."""
    model = tvm.resnet18(weights=None)
    model.conv1 = cnn_architectures.change_conv2d_channels(model.conv1, num_channels=1)
    model.fc = nn.Identity()
    model.constructor_name = "resnet18_1ch_embedder"
    return model


class Resnet18Classifier(nn.Module):
    """Mirror of upstream ``Resnet18_Classifier`` so the checkpoint state-dict loads."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.embedder = build_resnet18_1ch()
        self.classifier = nn.Linear(512, num_classes)
        self.constructor_name = "Resnet18_Classifier"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedder(x)
        logits = self.classifier(emb)
        return emb, logits


def load_ovenbird_extractor(device: torch.device) -> CNN:
    """Replacement for upstream ``load_ovenbird_model`` without network IO."""
    classifier = Resnet18Classifier(num_classes=NUM_CLASSES_PAPER)
    state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    cnn = CNN(classifier.embedder, sample_duration=2.0, classes=list(range(512)))
    preproc = OvenbirdPreprocessor()
    preproc.pipeline.load_audio.set(load_metadata=False)
    cnn.preprocessor = preproc
    cnn.network.embedding_layer = "avgpool"
    return cnn


def load_sample_table() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["file"] = df["file"].apply(lambda p: str((UPSTREAM / p.lstrip("./")).resolve()))
    return df.set_index(["file", "start_time", "end_time"])


def main() -> int:
    t_total = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    t_step = time.time()
    cnn = load_ovenbird_extractor(device)
    print(f"loaded feature extractor in {time.time() - t_step:.1f}s")

    table = load_sample_table()
    n_individuals = table["aiid_label"].nunique()
    print(f"sample: {len(table)} clips, {n_individuals} individuals")

    t_step = time.time()
    embeddings = cnn.embed(table)
    print(f"embeddings in {time.time() - t_step:.1f}s, shape={embeddings.shape}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = embeddings.copy()
    out.columns = [f"f{i:03d}" for i in range(out.shape[1])]
    out["aiid_label"] = table["aiid_label"].to_numpy()
    out.to_parquet(OUTPUT)
    print(f"saved embeddings to {OUTPUT.relative_to(ROOT)}")

    acc = leave_one_out_nn_accuracy(embeddings.to_numpy(), table["aiid_label"].to_numpy())
    print(f"sanity check (1-NN LOO, cosine): {acc:.3f}")

    print(f"total: {time.time() - t_total:.1f}s")
    return 0


def leave_one_out_nn_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 leave-one-out accuracy under cosine distance.

    Cheap correctness check: if the embedding extractor is loaded right, songs
    from the same individual should be each other's nearest neighbors.
    """
    distances = cosine_distances(features)
    np.fill_diagonal(distances, np.inf)
    nn_idx = distances.argmin(axis=1)
    return float((labels[nn_idx] == labels).mean())


if __name__ == "__main__":
    raise SystemExit(main())
