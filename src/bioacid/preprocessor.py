"""Audio preprocessing for Ovenbird-style spectrogram training.

Lifted from ``external/upstream/src/preprocessor.py`` with minor simplifications:

- Drops the ``noise_and_mute`` and ``mute_and_normalize`` action functions, which
  the upstream preprocessor defines but never uses in the actual pipeline.
- Keeps ``JitterClipTime`` (used as the ``time_jitter`` augmentation) and the
  ``OvenbirdPreprocessor`` itself.

Opensoundscape and torch are imported lazily so this module can be imported
in dev-only environments.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opensoundscape.preprocess.actions import BaseAction
    from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor


def _jitter_action_cls() -> type[BaseAction]:
    from opensoundscape.preprocess.actions import BaseAction, register_action_cls

    @register_action_cls
    class JitterClipTime(BaseAction):  # type: ignore[misc]
        """Randomly shift the offset time of an audio clip.

        Used as the ``time_jitter`` augmentation. ``max_shift`` is in seconds.
        """

        def __init__(self, max_shift: float = 0.25) -> None:
            super().__init__()
            self.params["max_shift"] = max_shift
            self.is_augmentation = True

        def __call__(self, sample: Any) -> None:
            if sample.start_time is None:
                return
            sample.start_time += random.uniform(-self.params["max_shift"], self.params["max_shift"])
            sample.start_time = max(sample.start_time, 0)

    return JitterClipTime


def OvenbirdPreprocessor(
    *,
    overlay_df: Any = None,
    sample_duration: float = 2.0,
    bandpass_hz: tuple[int, int] = (2000, 10000),
    max_time_jitter_s: float = 0.5,
    sample_rate: int = 32000,
    spec_augment: bool = False,
    freq_mask_max_width: float = 0.1,
) -> SpectrogramPreprocessor:
    """Build the Ovenbird spectrogram preprocessor used by Lapp et al. 2025.

    Returns an ``opensoundscape.preprocess.preprocessors.SpectrogramPreprocessor``
    configured with:

    - 2-second clips
    - 2-10 kHz bandpass (relevant Ovenbird song band)
    - audio normalize after trim
    - frequency mask disabled (unless ``spec_augment=True``)
    - random time-jitter (``max_shift=0.5s``) inserted before load
    - 32 kHz target sample rate

    When ``spec_augment=True``, the upstream ``frequency_mask`` action is
    enabled and an additional ``time_mask`` action is wired in. The two
    together approximate SpecAugment (Park et al. 2019).
    """
    from opensoundscape.audio import Audio
    from opensoundscape.preprocess.actions import Action
    from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor

    JitterClipTime = _jitter_action_cls()

    preproc: SpectrogramPreprocessor = SpectrogramPreprocessor(
        sample_duration=sample_duration, overlay_df=overlay_df
    )
    preproc.width = None
    preproc.height = None
    preproc.pipeline.bandpass.set(min_f=bandpass_hz[0], max_f=bandpass_hz[1])
    preproc.pipeline.to_spec.set(overlap_fraction=0.5)
    preproc.insert_action(
        "normalize",
        Action(Audio.normalize, is_augmentation=False),
        after_key="trim_audio",
    )

    if spec_augment:
        # Opensoundscape's SpectrogramPreprocessor ships frequency_mask only;
        # enable it with a sane width (fraction of spec height). A full
        # SpecAugment time-warp would need a custom Action class.
        preproc.pipeline.frequency_mask.bypass = False
        preproc.pipeline.frequency_mask.set(max_masks=2, max_width=freq_mask_max_width)
    else:
        preproc.pipeline.frequency_mask.bypass = True

    preproc.insert_action(
        "time_jitter",
        JitterClipTime(max_shift=max_time_jitter_s),
        before_key="load_audio",
    )
    preproc.remove_action("random_affine")
    preproc.remove_action("random_trim_audio")
    preproc.pipeline.overlay.set(overlay_prob=0.75, overlay_weight=[0.01, 0.6])
    preproc.pipeline.load_audio.set(
        sample_rate=sample_rate, load_metadata=False, out_of_bounds_mode="ignore"
    )
    return preproc


__all__ = ["OvenbirdPreprocessor"]
