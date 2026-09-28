"""
Preprocessing for the EEG Motor-Imagery pipeline (MNE-Python).

Pipeline (applied to each .bdf recording):
    load BDF  ->  pick EEG+EOG  ->  set montage  ->  detrend/DC-block
    ->  notch 50 Hz  ->  broadband band-pass  ->  (optional) ICA EOG removal
    ->  common-average reference (CAR)  ->  downsample
    ->  MI band-pass (8-30 Hz)  ->  epoching around Movement/Rest cues

The same parameters are reused online on the Pi (see online_inference.py), so
every step here is causal-friendly / cheap enough to run on an embedded device.
"""

from __future__ import annotations

import numpy as np
import mne
from scipy.signal import butter, sosfilt

from . import config as C

# Custom causal filter
def _causal_bandpass(data: np.ndarray, sfreq: float, l_freq: float,
                      h_freq: float, order: int = 4) -> np.ndarray:
    """Forward-only (causal) Butterworth band-pass — same as in online_inference.py."""
    sos = butter(order, [l_freq, h_freq], btype="band", fs=sfreq, output="sos")
    return sosfilt(sos, data, axis=-1)
    
# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path) -> mne.io.BaseRaw:
    """Read one BioSemi .bdf file and tag EEG / EOG channel types."""
    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")

    # First N_EEG channels are EEG; 65-68 are EOG. Everything else (leg EMG,
    # extra status) is dropped from the analysis set but EOG is kept for ICA.
    ch_names = raw.ch_names
    eeg_names = ch_names[: C.N_EEG]
    eog_names = [ch_names[i] for i in C.EOG_CHANNELS_IDX if i < len(ch_names)]

    type_map = {name: "eeg" for name in eeg_names}
    type_map.update({name: "eog" for name in eog_names})
    raw.set_channel_types(type_map)

    # Keep only EEG + EOG (+ the stim channel, which MNE handles separately).
    keep = eeg_names + eog_names
    if C.STIM_CHANNEL in raw.ch_names:
        keep = keep + [C.STIM_CHANNEL]
    raw.pick(keep)
    return raw

# --------------------------------------------------------------------------- #
# Continuous-signal cleaning
# --------------------------------------------------------------------------- #
def clean_continuous(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Detrend, notch, broadband filter, ICA-EOG, CAR, downsample (in place)."""
    # 1. Detrending / DC removal: a high-pass at the band-pass low edge removes
    #    slow drifts; we also subtract the per-channel mean first.
    raw.apply_function(lambda x: x - np.mean(x), picks="eeg")

    # 2. Notch out mains interference (and a harmonic if present).
    notch_freqs = [C.NOTCH_FREQ, 2 * C.NOTCH_FREQ]
    notch_freqs = [f for f in notch_freqs if f < raw.info["sfreq"] / 2]
    raw.notch_filter(notch_freqs, picks=["eeg", "eog"], verbose="ERROR")

    # 3. Broadband causal band-pass before ICA (1-40 Hz is the standard ICA input).
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    data = raw.get_data()
    data[picks, :] = _causal_bandpass(
        data[picks, :], raw.info["sfreq"], *C.BROAD_BANDPASS, order=4)
    raw._data[picks, :] = data[picks, :]

    # 4. ICA to remove ocular artifacts using the EOG channels as a reference. (Optional)
    if C.USE_ICA and len(raw.copy().pick("eog").ch_names) > 0:
        raw = _remove_eog_with_ica(raw)

    # 5. Common Average Reference (re-referencing).
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    # Or Surface Laplacian (Current Source Density)
    # raw = mne.preprocessing.compute_current_source_density(raw)
    
    # csd_channels = raw.copy().pick("csd").ch_names
    
    # # First changing to Volts (FIFF_UNIT_V) to be compatible with set_channel_types().
    # for ch_dict in raw.info["chs"]:
    #     if ch_dict["ch_name"] in csd_channels:
    #         ch_dict["unit"] = mne.io.constants.FIFF.FIFF_UNIT_V
            
    # # Setting as 'eeg' type
    # raw.set_channel_types({ch: "eeg" for ch in csd_channels})

    # 6. Downsample to lighten the load on the Pi.
    if raw.info["sfreq"] > C.SFREQ_TARGET:
        raw.resample(C.SFREQ_TARGET, verbose="ERROR")
    return raw


def _remove_eog_with_ica(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Fit ICA, auto-detect EOG-correlated components, and project them out."""
    ica = mne.preprocessing.ICA(
        n_components=C.ICA_N_COMPONENTS,
        method="fastica",
        random_state=C.RANDOM_STATE,
        max_iter="auto",
    )
    # Fit on a 1 Hz high-passed copy but apply to the real data.
    ica.fit(raw, picks="eeg", verbose="ERROR")
    try:
        eog_idx, _ = ica.find_bads_eog(raw, verbose="ERROR")
        ica.exclude = eog_idx
    except Exception as exc:
        print(f"[ICA] EOG detection failed ({exc}); no components removed.")
        ica.exclude = []
    ica.apply(raw, verbose="ERROR")
    return raw


# --------------------------------------------------------------------------- #
# Epoching
# --------------------------------------------------------------------------- #
def make_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """
    Band-pass into the MI band and cut epochs.

    Returns an mne.Epochs whose .events[:, 2] are remapped to 0/1 in the
    order (movement, rest) -> see config.EVENT_ID iteration order.
    """

    # MI-band filtering (mu+beta) — CAUSAL, same as the online path.
    raw_mi = raw.copy()
    eeg_picks = mne.pick_types(raw_mi.info, eeg=True)
    data = raw_mi.get_data()
    data[eeg_picks, :] = _causal_bandpass(
        data[eeg_picks, :], raw_mi.info["sfreq"], *C.BROAD_BANDPASS, order=4)
    raw_mi._data[eeg_picks, :] = data[eeg_picks, :]

    events = mne.find_events(raw_mi, stim_channel=C.STIM_CHANNEL,
                             verbose="ERROR")
    if events.size == 0:
        # Some BDF exports put triggers in annotations instead of a stim channel.
        events, _ = mne.events_from_annotations(raw_mi, verbose="ERROR")

    # Converting the EVENT_ID dictionary (that might be a list)
    # to a flat dictionary compatible with MNE (Trigger Code -> Name)
    flat_event_id = {}
    for name, codes in C.EVENT_ID.items():
        if isinstance(codes, list):
            for code in codes:
                flat_event_id[f"{name}/{code}"] = code
        else:
            flat_event_id[name] = codes

    # Adding a reject critiria for voltage over 100uV on a given epoch (instead of ICA)
    #reject_criteria = dict(eeg=100e-6)
    reject_criteria = None

    epochs = mne.Epochs(
        raw_mi, events, event_id=flat_event_id,
        tmin=C.TMIN, tmax=C.TMAX, baseline=C.BASELINE,
        picks="eeg", preload=True, reject=reject_criteria,
        reject_by_annotation=False, verbose="ERROR",
    )
    
    return epochs


# Build the preprocessing pipeline
def preprocess_file(path) -> mne.Epochs:
    """Full per-file pipeline: load -> montage -> clean -> epoch."""
    raw = load_raw(path)
    raw = clean_continuous(raw)
    epochs = make_epochs(raw)
    return epochs