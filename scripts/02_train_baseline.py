"""M2: Train a ResNet18 baseline on the sample dataset.

End-to-end smoke for the ported training pipeline: loads the sample CSV,
trains a randomly-initialised ResNet18 on the ``train`` split with supervised
cross-entropy, then embeds the full sample and evaluates clustering metrics.

Expectations: with only ~80 training clips covering 10 individuals, the
metrics will be **worse** than the upstream-checkpoint baseline (which was
trained on the full 234-individual PAM dataset). The goal here is to
demonstrate the ported pipeline runs cleanly end-to-end, not to match
upstream numbers.

Usage:
    uv run --extra ml --extra dev scripts/02_train_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bioacid.cluster import cluster_embeddings
from bioacid.data import AIIDLocalizedClipDataset, load_clip_table
from bioacid.evaluate import clustering_metrics, format_metrics
from bioacid.preprocessor import OvenbirdPreprocessor
from bioacid.train import TrainConfig, train_supervised

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "upstream"
CSV_PATH = UPSTREAM / "sample_data" / "labeled_clips_sample.csv"
RUN_DIR = ROOT / "reports" / "runs" / "m2_baseline"


def main() -> int:
    if not UPSTREAM.exists():
        print(f"upstream not found at {UPSTREAM}; clone first", file=sys.stderr)
        return 1

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = load_clip_table(CSV_PATH, audio_root=UPSTREAM, set_index=False)
    print(f"loaded {len(df)} clips, splits={df['data_split'].value_counts().to_dict()}")

    train_df = df[df["data_split"] == "train"].copy()
    unique_labels = sorted(train_df["aiid_label"].unique().tolist())
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    print(f"training on {len(train_df)} clips, {len(unique_labels)} individuals")

    preproc = OvenbirdPreprocessor()
    model, history = train_supervised(
        train_df,
        preprocessor=preproc,
        num_classes=len(unique_labels),
        label_to_index=label_to_index,
        config=TrainConfig(epochs=10, batch_size=32, lr=1e-3, num_workers=0),
        device=device,
    )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RUN_DIR / "model.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"checkpoint saved to {ckpt_path.relative_to(ROOT)}")

    print("\nembedding full sample...")
    embeddings = _embed_all(model, df, preproc, device)
    metrics = _cluster_and_eval(embeddings, df["aiid_label"].to_numpy())
    print(format_metrics(metrics))

    history_path = RUN_DIR / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "epoch_loss": history.epoch_loss,
                "epoch_top1": history.epoch_top1,
                "metrics": metrics.as_dict(),
            },
            indent=2,
        )
    )
    print(f"\ntotal: {time.time() - t0:.1f}s, run artifacts in {RUN_DIR.relative_to(ROOT)}")
    return 0


def _embed_all(
    model: torch.nn.Module, df: pd.DataFrame, preproc: object, device: torch.device
) -> np.ndarray:
    from torch.utils.data import DataLoader

    dataset = AIIDLocalizedClipDataset(df, preprocessor=preproc, bypass_augmentations=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda b: b)
    model.eval()
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = torch.vstack([s.data[None, :, :] for s in batch]).to(device)
            emb, _ = model(x)
            feats.append(emb.detach().cpu().numpy())
    return np.vstack(feats)


def _cluster_and_eval(embeddings: np.ndarray, truth: np.ndarray) -> object:
    predicted, _ = cluster_embeddings(
        embeddings,
        reduction_algorithm="umap",
        reduced_n_dimensions=5,
        min_cluster_size=5,
        random_state=42,
    )
    return clustering_metrics(truth, predicted)


if __name__ == "__main__":
    raise SystemExit(main())
