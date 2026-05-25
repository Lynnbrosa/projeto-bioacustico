"""Single-config experiment runner for the M3 grid.

Provides :func:`run_experiment` that takes a :class:`bioacid.train.TrainConfig`,
trains a backbone + loss-head on the public sample's training split, embeds
the full sample, runs UMAP + HDBSCAN, and returns the clustering metrics.

The harness keeps the contract narrow on purpose: one config in, one
metrics record out. Looping over configs and aggregating is the caller's job
(see ``scripts/04_run_experiments.py``).

Heavy imports are kept lazy so this module imports cleanly without ML extras.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from bioacid.evaluate import ClusteringMetrics
from bioacid.train import TrainConfig


@dataclass
class ExperimentResult:
    """Outcome of a single experiment run."""

    config: TrainConfig
    metrics: ClusteringMetrics
    train_seconds: float
    embed_seconds: float
    final_train_loss: float
    final_train_top1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "metrics": self.metrics.as_dict(),
            "train_seconds": self.train_seconds,
            "embed_seconds": self.embed_seconds,
            "final_train_loss": self.final_train_loss,
            "final_train_top1": self.final_train_top1,
        }


def run_experiment(
    config: TrainConfig,
    *,
    train_csv: str | Path,
    audio_root: str | Path,
    train_split: str = "val",
    output_dir: Path | None = None,
    device: str = "cpu",
) -> ExperimentResult:
    """Run one (backbone, loss) configuration end-to-end on the public sample.

    Because the sample CSV has no ``train`` split (it splits into ``val``
    and ``test``), the trainer learns from ``train_split`` (default ``val``)
    and clustering metrics are computed on all clips combined — matching the
    upstream demo evaluation. Open-set in spirit: ``test`` individuals are
    not seen during backbone training.
    """
    import numpy as np
    import torch

    from bioacid.cluster import cluster_embeddings
    from bioacid.data import load_clip_table
    from bioacid.evaluate import clustering_metrics
    from bioacid.preprocessor import OvenbirdPreprocessor
    from bioacid.train import train_supervised

    full_df = load_clip_table(train_csv, audio_root=audio_root, set_index=False)
    train_df = full_df[full_df["data_split"] == train_split].copy()
    unique_labels = sorted(train_df["aiid_label"].unique().tolist())
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    preproc = OvenbirdPreprocessor()

    t_train = time.time()
    backbone, _, history = train_supervised(
        train_df,
        preprocessor=preproc,
        num_classes=len(unique_labels),
        label_to_index=label_to_index,
        config=config,
        device=device,
    )
    train_seconds = time.time() - t_train

    t_embed = time.time()
    embeddings = _embed(backbone, full_df, preproc, device=device)
    embed_seconds = time.time() - t_embed

    predicted, _ = cluster_embeddings(
        embeddings,
        reduction_algorithm="umap",
        reduced_n_dimensions=5,
        min_cluster_size=5,
        random_state=42,
    )
    metrics = clustering_metrics(full_df["aiid_label"].to_numpy(), predicted)

    result = ExperimentResult(
        config=config,
        metrics=metrics,
        train_seconds=train_seconds,
        embed_seconds=embed_seconds,
        final_train_loss=history.epoch_loss[-1] if history.epoch_loss else float("nan"),
        final_train_top1=history.epoch_top1[-1] if history.epoch_top1 else float("nan"),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(json.dumps(result.as_dict(), indent=2))
        torch.save(backbone.state_dict(), output_dir / "backbone.pth")
        np.save(output_dir / "embeddings.npy", embeddings)

    return result


def _embed(backbone: Any, df: Any, preproc: Any, *, device: str) -> Any:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from bioacid.data import AIIDLocalizedClipDataset

    dataset = AIIDLocalizedClipDataset(df, preprocessor=preproc, bypass_augmentations=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda b: b)
    backbone.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = torch.vstack([s.data[None, :, :] for s in batch]).to(device)
            features = backbone(x)
            chunks.append(features.detach().cpu().numpy())
    return np.vstack(chunks)


__all__ = ["ExperimentResult", "run_experiment"]
