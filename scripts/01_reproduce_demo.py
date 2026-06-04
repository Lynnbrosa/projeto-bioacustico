"""M1: Reproduce the upstream demo on the public Ovenbird sample dataset.

Loads the pretrained Ovenbird feature extractor checkpoint, generates
embeddings for the 100 sample clips (10 individuals) and saves them as a
parquet artifact under ``data/processed/sample_embeddings.parquet``.

Uses :func:`bioacid.models.load_ovenbird_checkpoint` (our ported loader) and
:func:`bioacid.data.load_clip_table` for the CSV.

Usage:
    uv run --extra ml --extra dev scripts/01_reproduce_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances

from bioacid.data import load_clip_table
from bioacid.models import load_ovenbird_checkpoint

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "upstream"
CHECKPOINTS_DIR = UPSTREAM / "checkpoints"
CSV_PATH = UPSTREAM / "sample_data" / "labeled_clips_sample.csv"
OUTPUT = ROOT / "data" / "processed" / "sample_embeddings.parquet"

# The upstream checkpoint ships as ``full_2025-04-10T11:02:36.028451_best.pth``.
# Windows can't materialise that filename (``:`` is illegal), so on Windows users
# typically download it as ``ovenbird_best.pth`` (or similar). We auto-detect any
# ``.pth`` in the checkpoints dir to stay portable.
_PREFERRED_CHECKPOINTS = (
    "full_2025-04-10T11:02:36.028451_best.pth",
    "ovenbird_best.pth",
)


def find_checkpoint() -> Path | None:
    for name in _PREFERRED_CHECKPOINTS:
        candidate = CHECKPOINTS_DIR / name
        if candidate.exists():
            return candidate
    fallback = sorted(CHECKPOINTS_DIR.glob("*.pth"))
    return fallback[0] if fallback else None


def main() -> int:
    if not UPSTREAM.exists():
        print(
            f"Upstream repo not found at {UPSTREAM}. Clone it with:\n"
            "  git clone https://github.com/sammlapp/ovenbird-individual-recognition.git "
            f"{UPSTREAM}",
            file=sys.stderr,
        )
        return 1

    checkpoint = find_checkpoint()
    if checkpoint is None:
        print(
            f"No checkpoint found under {CHECKPOINTS_DIR}.\n"
            "On Windows the upstream file fails to checkout (colon in filename); "
            "download it manually with:\n"
            '  curl -L "https://github.com/sammlapp/ovenbird-individual-recognition/raw/main/'
            'checkpoints/full_2025-04-10T11%3A02%3A36.028451_best.pth" '
            "-o external/upstream/checkpoints/ovenbird_best.pth",
            file=sys.stderr,
        )
        return 1

    t_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint.name}")

    t_step = time.time()
    cnn = load_ovenbird_checkpoint(checkpoint, device=device)
    print(f"loaded feature extractor in {time.time() - t_step:.1f}s")

    table = load_clip_table(CSV_PATH, audio_root=UPSTREAM)
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

    features = embeddings.to_numpy()
    labels = table["aiid_label"].to_numpy()
    acc = leave_one_out_nn_accuracy(features, labels)
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
