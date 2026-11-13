"""
Real-time inference for the Raspberry Pi.

It loads a model bundle saved by train.py and classifies sliding windows of
incoming EEG. The window preprocessing mirrors the offline pipeline so the
features the classifier sees online match what it was trained on.

Changes vs the previous version (all opt-in, zero behavioural change when
`timer=None`):
  * `push()` and `try_predict()` accept a `StageTimer`; every preprocessing
    step and every sklearn pipeline step is timed separately, so the receiver
    can report the share of the per-prediction cost spent in the filter bank,
    the feature selection, the PCA and the classifier itself.
  * the debug prints were removed from `try_predict()`: a `print()` inside the
    timed region costs milliseconds (much more over SSH) and was being counted
    as inference latency. Scores are now returned, and printing is the
    caller's job, outside the timer.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import joblib
from scipy.signal import butter, iirnotch, sosfilt, tf2sos, sosfilt_zi

from . import config as C


def _track(timer, name):
    """`with _track(timer, "stage"):` -- a no-op when timer is None."""
    return timer.track(name) if timer is not None else nullcontext()


class RealTimeMI:
    def __init__(self, bundle_path: str, all_ch_names: list[str] | None = None,
                 sfreq_in: float = C.SFREQ_ORIG,
                 apply_mi_bandpass: bool = True):
        self.b = joblib.load(bundle_path)
        self.apply_mi_bandpass = apply_mi_bandpass
        self.pipeline = self.b["pipeline"]
        self.model_channels = self.b["channels"]
        self.model_name = self.b.get("model_name", "unknown")
        self.sfreq_in = sfreq_in
        self.sfreq = self.b["sfreq"]
        self.win_sec = C.WINDOW_LEN
        self.win_samples = int(round(self.win_sec * self.sfreq))
        self.decim = max(1, int(round(sfreq_in / self.sfreq)))

        self.sample_count = 0

        if all_ch_names is not None:
            self.ch_idx = [all_ch_names.index(c) for c in self.model_channels]
        else:
            self.ch_idx = list(range(len(self.model_channels)))

        # Designing IIR filters for the initial sampling frequency (512 Hz)
        ny = self.sfreq_in / 2
        lo, hi = self.b["bandpass"]
        self.sos_bp = butter(4, [lo / ny, hi / ny], btype="band", output="sos")
        bn, an = iirnotch(C.NOTCH_FREQ / ny, Q=30)
        self.sos_notch = tf2sos(bn, an)

        # MI band-pass (3-30 Hz) at the model rate, same as make_epochs().
        # Offline it is a causal sosfilt with ZERO initial state over the whole
        # recording, so the state starts at zero here too and is carried across
        # chunks. It runs on the motor channels only (per-channel linear filter,
        # so picking before/after is equivalent).
        mi_lo, mi_hi = self.b.get("mi_bandpass", C.BANDPASS)
        self.sos_mi = butter(4, [mi_lo, mi_hi], btype="band", fs=self.sfreq,
                             output="sos")
        self.zi_mi = np.zeros((self.sos_mi.shape[0], len(self.ch_idx), 2))

        # Tracing filter states for all 64 channels to ensure correct CAR
        self.zi_notch = None
        self.zi_bp = None
        self._buf = np.zeros((len(self.ch_idx), 0))

        # Total decimated (128 Hz) samples ever produced. The receiver should
        # gate decisions on THIS, not on time.time(): wall-clock gating makes
        # the number of decisions per trial depend on OS/LSL scheduling
        # jitter, so the same recording produces a different number of
        # windows per trial on every run. Counting emitted samples makes the
        # window grid deterministic and tied only to the data itself.
        self.n_produced = 0

        self.labels = list(self.b["event_id"].keys())

    # ----------------------------------------------------------------- #
    @property
    def ready(self) -> bool:
        """True once the ring buffer holds a full decision window."""
        return self._buf.shape[1] >= self.win_samples

    def warm_up(self, n_calls: int = 3) -> None:
        """
        Run a few dummy predictions on random data so that lazy imports, BLAS
        thread pools and first-touch page faults are NOT charged to the first
        real decision. Filter states are untouched (we only touch the pipeline).
        """
        X = np.random.randn(1, len(self.ch_idx), self.win_samples) * 1e-5
        for _ in range(max(1, n_calls)):
            try:
                self.pipeline.predict(X)
            except Exception:
                return

    # ----------------------------------------------------------------- #
    def push(self, samples: np.ndarray, timer=None):
        """samples: (n_all_channels, n_new) at the ORIGINAL sampling rate (512 Hz)."""

        # 1. Isolating only the 64 EEG channels
        with _track(timer, "pre/select_eeg"):
            x_eeg = samples[:C.N_EEG, :]
            if x_eeg.dtype != np.float64:
                x_eeg = x_eeg.astype(np.float64, copy=True)
            else:
                x_eeg = x_eeg.copy()
        n_new = x_eeg.shape[1]
        if n_new == 0:
            return

        # 2. Initializing IIR states based on the 1st sample.
        # This practice neutralizes DC offsets of the signal without transients
        if self.zi_notch is None:
            zi_n = sosfilt_zi(self.sos_notch)
            zi_b = sosfilt_zi(self.sos_bp)
            x0 = x_eeg[:, 0]
            self.zi_notch = zi_n[:, np.newaxis, :] * x0[np.newaxis, :, np.newaxis]
            self.zi_bp = zi_b[:, np.newaxis, :] * x0[np.newaxis, :, np.newaxis]

        # 3. Filtering all EEG channels while in 512 Hz
        with _track(timer, "pre/notch_50hz"):
            x_filt, self.zi_notch = sosfilt(self.sos_notch, x_eeg, axis=-1,
                                            zi=self.zi_notch)
        with _track(timer, "pre/bandpass"):
            x_filt, self.zi_bp = sosfilt(self.sos_bp, x_filt, axis=-1,
                                         zi=self.zi_bp)

        # 4. Common Average Reference over all filtered EEG channels
        with _track(timer, "pre/car"):
            x_car = x_filt - x_filt.mean(axis=0, keepdims=True)

        # 5. Isolating the configured motor channels
        with _track(timer, "pre/pick_motor"):
            x_motor = x_car[self.ch_idx, :]

        # 6. Downsampling to 128 Hz (maintaining phase)
        with _track(timer, "pre/decimate"):
            if self.decim > 1:
                offset = (self.decim - (self.sample_count % self.decim)) % self.decim
                x_motor = x_motor[:, offset::self.decim]
                self.sample_count += n_new

        # 7. MI band-pass at 128 Hz (the step that was missing online)
        with _track(timer, "pre/mi_bandpass"):
            if self.apply_mi_bandpass and x_motor.shape[1] > 0:
                x_motor, self.zi_mi = sosfilt(self.sos_mi, x_motor, axis=-1,
                                              zi=self.zi_mi)

        # 8. Ring buffer update
        with _track(timer, "pre/buffer"):
            self._buf = np.concatenate([self._buf, x_motor], axis=1)
            self.n_produced += x_motor.shape[1]
            maxlen = 2 * self.win_samples
            if self._buf.shape[1] > maxlen:
                self._buf = self._buf[:, -maxlen:]

    # ----------------------------------------------------------------- #
    def try_predict(self, timer=None, return_scores: bool = False):
        """
        Classify the most recent window.

        Returns the label string, or (label, scores) when return_scores=True.
        With a StageTimer attached, every sklearn pipeline step is timed
        individually under the "predict/<step_name>" key.
        """
        if not self.ready:
            return (None, None) if return_scores else None

        with _track(timer, "predict/window_copy"):
            win = self._buf[:, -self.win_samples:]
            X = np.ascontiguousarray(win)[np.newaxis, :, :]

        steps = getattr(self.pipeline, "steps", None)
        if timer is None or steps is None:
            idx = int(self.pipeline.predict(X)[0])
            scores = self._scores(X) if return_scores else None
        else:
            Xt = X
            for name, step in steps[:-1]:
                with timer.track(f"predict/{name}"):
                    Xt = step.transform(Xt)
            name, clf = steps[-1]
            with timer.track(f"predict/{name}"):
                idx = int(clf.predict(Xt)[0])
            scores = None
            if return_scores:
                scores = self._scores_from_features(clf, Xt)

        label = self.labels[idx]
        return (label, scores) if return_scores else label

    # ----------------------------------------------------------------- #
    def _scores(self, X):
        try:
            if hasattr(self.pipeline, "predict_proba"):
                return np.asarray(self.pipeline.predict_proba(X)[0]).tolist()
            if hasattr(self.pipeline, "decision_function"):
                return np.atleast_1d(self.pipeline.decision_function(X)[0]).tolist()
        except Exception:
            pass
        return None

    @staticmethod
    def _scores_from_features(clf, Xt):
        try:
            if hasattr(clf, "predict_proba"):
                return np.asarray(clf.predict_proba(Xt)[0]).tolist()
            if hasattr(clf, "decision_function"):
                return np.atleast_1d(clf.decision_function(Xt)[0]).tolist()
        except Exception:
            pass
        return None