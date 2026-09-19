"""
Real-time inference skeleton for the Raspberry Pi.

It loads a model bundle saved by train.py and classifies sliding windows of
incoming EEG. The window preprocessing mirrors the offline pipeline so the
features the classifier sees online match what it was trained on.

"""

from __future__ import annotations

import numpy as np
import joblib
from scipy.signal import butter, iirnotch, sosfilt, tf2sos, sosfilt_zi

from . import config as C

class RealTimeMI:
    def __init__(self, bundle_path: str, all_ch_names: list[str] | None = None,
                 sfreq_in: float = C.SFREQ_ORIG):
        self.b = joblib.load(bundle_path)
        self.pipeline = self.b["pipeline"]
        self.model_channels = self.b["channels"]
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

        # Tracing filter states for total 64 channels to ensure correct CAR
        self.zi_notch = None
        self.zi_bp = None
        self._buf = np.zeros((len(self.ch_idx), 0))

    def push(self, samples: np.ndarray):
        """samples: (n_all_channels, n_new) at the ORIGINAL sampling rate (512 Hz)."""
        
        # 1. Isolating only the 64 EEG channels
        x_eeg = samples[:C.N_EEG, :].copy()
        n_new = x_eeg.shape[1]
        if n_new == 0:
            return

        # 2. Initializing IIR states based on the 1st sample.
        # This practice neutralizes DC offsets of the signal without transients
        if self.zi_notch is None:
            zi_n = sosfilt_zi(self.sos_notch)
            zi_b = sosfilt_zi(self.sos_bp)
            x0 = x_eeg[:, 0]
            # Fitting zi state to all EEG channels
            self.zi_notch = zi_n[:, np.newaxis, :] * x0[np.newaxis, :, np.newaxis]
            self.zi_bp = zi_b[:, np.newaxis, :] * x0[np.newaxis, :, np.newaxis]

        # 3. Filtering all EEG channels while in 512 Hz (Notch 50Hz + Bandpass 1-40Hz)
        x_filt, self.zi_notch = sosfilt(self.sos_notch, x_eeg, axis=-1, zi=self.zi_notch)
        x_filt, self.zi_bp = sosfilt(self.sos_bp, x_filt, axis=-1, zi=self.zi_bp)

        # 4. Implementing CAR to all EEG filterd channels
        car_mean = x_filt.mean(axis=0, keepdims=True)
        x_car = x_filt - car_mean

        # 5. Isolating the motor channels that are configured
        x_motor = x_car[self.ch_idx, :]

        # 6. Downsampling to 128 Hz (maintaining phase)
        if self.decim > 1:
            offset = (self.decim - (self.sample_count % self.decim)) % self.decim
            x_motor = x_motor[:, offset::self.decim]
            self.sample_count += n_new

        self._buf = np.concatenate([self._buf, x_motor], axis=1)
        
        maxlen = 2 * self.win_samples
        if self._buf.shape[1] > maxlen:
            self._buf = self._buf[:, -maxlen:]

    def try_predict(self):
        if self._buf.shape[1] < self.win_samples:
            return None
        win = self._buf[:, -self.win_samples:].copy()
        # For debugging: print window's signal statistics
        # print(f"[ONLINE] mean={win.mean():.3e} std={win.std():.3e} "
        #      f"min={win.min():.3e} max={win.max():.3e}")
        X = win[np.newaxis, :, :]
        idx = int(self.pipeline.predict(X)[0])
        # For debugging: print decision scores
        if hasattr(self.pipeline, "decision_function"):
            print(f"  decision_function: {self.pipeline.decision_function(X)}")
        elif hasattr(self.pipeline, "predict_proba"):
            print(f"  predict_proba: {self.pipeline.predict_proba(X)}")
        labels = list(self.b["event_id"].keys())
        return labels[idx]