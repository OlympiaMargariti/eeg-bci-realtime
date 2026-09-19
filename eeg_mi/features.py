"""
Channel selection + feature extraction.

Two feature representations are produced from the same cleaned epochs:

  * For SVM / Random-Forest  -> CSP spatial filtering + log-variance band power.
        Channels are first reduced to a small motor-cortex set, then CSP learns
        spatial filters that maximise the variance ratio between classes; the
        log-variance of the filtered signals is a compact, classifier-friendly
        feature (the standard MI feature).

  * For Riemannian MDM       -> trial covariance matrices (no feature vector).
        MDM classifies directly on the SPD covariance manifold, so
        we feed it the covariances of the selected channels.
"""

from __future__ import annotations

import numpy as np
import mne

from . import config as C

def select_motor_channels(epochs: mne.Epochs) -> mne.Epochs:
    """
    Keep only the sensorimotor channels in config.MOTOR_CHANNELS_1020.

    Resolution order (most reliable first):
      1) the recording already uses 10-20 names  -> match by name
      2) the recording uses BioSemi A/B names     -> translate via lookup
      3) neither                                  -> nearest montage position
    """
    wanted = C.MOTOR_CHANNELS_1020
    eeg_names = epochs.copy().pick("eeg").ch_names
    lower = {n.lower(): n for n in eeg_names}

    # 1) direct 10-20 name match (case-insensitive)
    chosen = [lower[w.lower()] for w in wanted if w.lower() in lower]

    # 2) position fallback
    if len(chosen) < max(3, len(wanted) // 2):
        chosen = _nearest_by_position(epochs, wanted)

    chosen = list(dict.fromkeys(chosen))                 # dedupe, keep order
    if not chosen:
        raise RuntimeError("Could not resolve any motor channel.")
    print(f"[channels] using {chosen} for {wanted}")
    return epochs.copy().pick(chosen)


def _nearest_by_position(epochs: mne.Epochs, wanted) -> list[str]:
    ref_pos = mne.channels.make_standard_montage("standard_1020") \
        .get_positions()["ch_pos"]
    dig = {ch["ch_name"]: ch["loc"][:3] for ch in epochs.info["chs"]
           if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH
           and not np.allclose(ch["loc"][:3], 0)}
    if not dig:
        return []
    out = []
    for w in wanted:
        if w in ref_pos:
            out.append(min(dig, key=lambda c: np.linalg.norm(dig[c] - ref_pos[w])))
    return out