"""Datasets and table helpers for bioacoustic individual identification.

Ports the minimum from upstream's ``dataset.py`` needed for inference and
basic supervised training on the public sample.

The richer training-time samplers and pseudo-label dataset variants from
upstream are out of scope for M2 and will be added during M3.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from torch.utils.data import Dataset


def load_clip_table(
    csv_path: str | Path,
    *,
    audio_root: str | Path | None = None,
    set_index: bool = True,
) -> pd.DataFrame:
    """Load a labeled-clips CSV and optionally rewrite ``file`` paths.

    The upstream sample CSV stores ``file`` as ``./sample_data/audio_clips/...``
    relative to the upstream repo root. ``audio_root`` lets callers point at a
    different filesystem location without editing the CSV.

    When ``set_index`` is true (default) the dataframe is indexed by
    ``(file, start_time, end_time)`` — the format expected by
    ``opensoundscape.CNN.embed``.
    """
    df = pd.read_csv(csv_path)
    if audio_root is not None:
        root = Path(audio_root).resolve()
        df["file"] = df["file"].apply(lambda p: str((root / str(p).lstrip("./")).resolve()))
    if set_index and {"file", "start_time", "end_time"}.issubset(df.columns):
        df = df.set_index(["file", "start_time", "end_time"])
    return df


def AIIDLocalizedClipDataset(
    aiid_df: pd.DataFrame,
    preprocessor: Any,
    *,
    bypass_augmentations: bool = False,
    unique_labels: list[int] | None = None,
) -> Dataset[Any]:
    """Build the upstream ``AIIDLocalizedClipDataset`` lazily.

    Returns a torch ``Dataset`` whose ``__getitem__`` yields preprocessed
    samples ready to be batched into a ``DataLoader``.

    Ported from ``upstream/src/dataset.py``. Kept as a factory wrapper so the
    bioacid package imports cleanly without torch/opensoundscape installed.
    """
    import numpy as np
    import pandas as pd
    from opensoundscape.annotations import categorical_to_multi_hot
    from opensoundscape.sample import AudioSample
    from torch.utils.data import Dataset

    class _AIIDLocalizedClipDataset(Dataset[Any]):
        def __init__(
            self,
            df: pd.DataFrame,
            preprocessor: Any,
            bypass_augmentations: bool,
            unique_labels: list[int] | None,
        ) -> None:
            self.aiid_df = df.reset_index(drop=True)
            self.preprocessor = preprocessor
            self.bypass_augmentations = bypass_augmentations

            if unique_labels is None and "aiid_label" in self.aiid_df.columns:
                unique_labels_arr = self.aiid_df["aiid_label"].unique()
                if not all(isinstance(label, int | np.integer) for label in unique_labels_arr):
                    raise ValueError("Labels must be integers.")
                unique_labels = list(unique_labels_arr)
            self.aiid_label_list = unique_labels

            if "pseudo_label" not in self.aiid_df.columns:
                self.aiid_df["pseudo_label"] = -1

            clip_duration = self.preprocessor.sample_duration
            start_times = self.aiid_df.song_center_time - clip_duration / 2
            end_times = start_times + clip_duration
            index = pd.DataFrame(
                {
                    "file": self.aiid_df.file,
                    "start_time": start_times,
                    "end_time": end_times,
                }
            ).set_index(["file", "start_time", "end_time"])
            if "aiid_label" in self.aiid_df.columns and unique_labels is not None:
                multihot_sp, _ = categorical_to_multi_hot(
                    [[a] for a in self.aiid_df["aiid_label"].values],
                    unique_labels,
                    sparse=True,
                )
                self.label_df = pd.DataFrame.sparse.from_spmatrix(  # type: ignore[attr-defined]
                    multihot_sp, index=index.index, columns=unique_labels
                )
            else:
                self.label_df = pd.DataFrame(index=index.index)

        def __getitem__(self, idx: int) -> Any:
            if not isinstance(idx, int):
                raise TypeError(f"idx must be an integer, got {type(idx)}")
            sample = AudioSample.from_series(self.label_df.iloc[idx])
            sample = self.preprocessor.forward(
                sample, bypass_augmentations=self.bypass_augmentations
            )
            sample.idx = idx
            for col in ("array", "event_id", "pseudo_label", "aiid_label"):
                if col in self.aiid_df.columns:
                    setattr(sample, col, self.aiid_df.iloc[idx][col])
            return sample

        def __len__(self) -> int:
            return len(self.label_df)

    return _AIIDLocalizedClipDataset(
        df=aiid_df,
        preprocessor=preprocessor,
        bypass_augmentations=bypass_augmentations,
        unique_labels=unique_labels,
    )


__all__ = ["AIIDLocalizedClipDataset", "load_clip_table"]
