"""M4: Apply the bioacid pipeline to neotropical recordings.

Reads the metadata.json produced by ``scripts/04_xeno_canto_pull.py``,
segments each recording into 2s clips, generates embeddings using a
ResNet18 backbone (random init or pre-trained Ovenbird checkpoint), runs
HDBSCAN, and reports clustering metrics against the recordist+date+locality
pseudo-labels.

Usage:
    uv run --extra ml --extra dev scripts/05_extend_to_species.py \\
        --metadata data/raw/xenocanto/turdus_rufiventris/metadata.json \\
        --pretrained-checkpoint external/upstream/checkpoints/full_2025-04-10T11:02:36.028451_best.pth

Limitations
-----------
- Without an Ovenbird-style pre-trained extractor for the target species,
  embeddings inherit biases from the source species. Best practice is to
  fine-tune on a small curated subset (M4 follow-up).
- Pseudo-labels are noisy by construction (see report).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Defaults to the metadata file's parent directory.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Ovenbird ResNet18 checkpoint. If omitted, uses random init.",
    )
    parser.add_argument("--clip-duration", type=float, default=2.0)
    parser.add_argument("--clip-stride", type=float, default=2.0)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "runs" / "m4_extension",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.metadata.exists():
        print(f"metadata not found: {args.metadata}", file=sys.stderr)
        print("Run scripts/04_xeno_canto_pull.py first.", file=sys.stderr)
        return 1

    import numpy as np
    import torch
    from opensoundscape import CNN

    from bioacid.cluster import cluster_embeddings
    from bioacid.evaluate import clustering_metrics, format_metrics
    from bioacid.models import build_resnet18_1ch
    from bioacid.preprocessor import OvenbirdPreprocessor

    metadata = json.loads(args.metadata.read_text())
    audio_dir = args.audio_dir or args.metadata.parent

    clip_table = _build_clip_table(
        recordings=metadata["recordings"],
        audio_dir=audio_dir,
        clip_duration=args.clip_duration,
        clip_stride=args.clip_stride,
    )
    if clip_table.empty:
        print("no clips generated (check that audio files exist)", file=sys.stderr)
        return 2
    print(
        f"prepared {len(clip_table)} clips from {clip_table['file'].nunique()} recordings, "
        f"{clip_table['pseudo_individual_id'].nunique()} pseudo-individuals"
    )

    backbone = build_resnet18_1ch()
    if args.pretrained_checkpoint is not None and args.pretrained_checkpoint.exists():
        print(f"loading pre-trained extractor from {args.pretrained_checkpoint}")
        from bioacid.models import Resnet18Classifier

        classifier = Resnet18Classifier(num_classes=234)
        state_dict = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=True)
        classifier.load_state_dict(state_dict)
        backbone = classifier.embedder  # type: ignore[attr-defined]

    preproc = OvenbirdPreprocessor(sample_duration=args.clip_duration)
    cnn = CNN(backbone, sample_duration=args.clip_duration, classes=list(range(512)))
    cnn.preprocessor = preproc
    cnn.network.embedding_layer = "avgpool"

    t0 = time.time()
    embeddings = cnn.embed(clip_table.set_index(["file", "start_time", "end_time"]))
    print(f"embeddings shape={embeddings.shape}, time={time.time() - t0:.1f}s")

    predicted, _ = cluster_embeddings(
        np.asarray(embeddings.to_numpy(), dtype=np.float32),
        reduction_algorithm="umap",
        reduced_n_dimensions=5,
        min_cluster_size=args.min_cluster_size,
        random_state=42,
    )
    truth = clip_table["pseudo_individual_id"].to_numpy()
    metrics = clustering_metrics(truth, predicted)

    print()
    print(format_metrics(metrics))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "n_clips": len(clip_table),
        "n_recordings": int(clip_table["file"].nunique()),
        "n_pseudo_individuals": int(clip_table["pseudo_individual_id"].nunique()),
        "n_predicted_clusters": len(np.unique(predicted[predicted >= 0])),
        "metrics_vs_pseudo_labels": metrics.as_dict(),
    }
    (args.output_dir / "result.json").write_text(json.dumps(out, indent=2))
    print(f"\nresult: {(args.output_dir / 'result.json').relative_to(ROOT)}")
    return 0


def _build_clip_table(
    *,
    recordings: Iterable[dict[str, object]],
    audio_dir: Path,
    clip_duration: float,
    clip_stride: float,
) -> pd.DataFrame:
    import pandas as pd

    rows: list[dict[str, object]] = []
    for rec in recordings:
        audio_path = audio_dir / f"{rec['id']}.mp3"
        if not audio_path.exists():
            continue
        length_s = float(rec.get("length_s", 0.0))
        if length_s < clip_duration:
            continue
        for start in _frange(0.0, length_s - clip_duration, clip_stride):
            rows.append(
                {
                    "file": str(audio_path.resolve()),
                    "start_time": start,
                    "end_time": start + clip_duration,
                    "pseudo_individual_id": int(rec["pseudo_individual_id"]),
                    "recording_id": rec["id"],
                }
            )
    return pd.DataFrame(rows)


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    while value <= stop + 1e-9:
        yield round(value, 6)
        value += step


if __name__ == "__main__":
    raise SystemExit(main())
