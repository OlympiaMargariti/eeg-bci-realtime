"""
Step-0 "replay" test: does the ONLINE preprocessing reproduce the OFFLINE one?

No LSL, no wall clock, no ground-truth channel. The recording of one subject
is pushed through `RealTimeMI.push()` in small chunks exactly like the
streamer/receiver would do, and the resulting 128 Hz motor-channel stream is
compared with what `train.py` stored in the bundle:

  1. SIGNAL LEVEL  -- for every test trial, the online samples of the epoch
     [cue+TMIN, cue+TMAX] vs the offline epoch saved in the bundle (X_test):
     relative RMS error and per-channel correlation, plus a +-N sample
     alignment search (a constant shift != 0 means a timing convention bug).
  2. PREDICTION LEVEL -- the same fitted pipeline classifies both versions:
     agreement rate (online vs offline) and accuracy of each against y_test.
  3. ALIGNMENT SENSITIVITY -- accuracy when the online window ends earlier/
     later than the nominal cue+TMAX. This is what a free-running sliding
     window does to a model trained on one fixed alignment.

If (1) and (2) are clean, any later accuracy loss in the LSL receiver comes
from window timing / scoring, not from preprocessing.

Run (from the folder that contains the eeg_mi package):
    python -m eeg_mi.replay_test -s 1 -m 3
    python -m eeg_mi.replay_test -s 1 --all-models
    python -m eeg_mi.replay_test -s 1 -m 3 --legacy     # before/after of the 3-30 Hz fix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from . import config as C
from .online_inference import RealTimeMI

# Heuristic thresholds for the PASS/CHECK verdict (not a standard, tune freely)
PASS_REL_ERR = 0.10       # median relative RMS error of the epoch
PASS_AGREEMENT = 0.95     # online/offline prediction agreement


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
def _ci_find(directory: Path, name: str) -> Path:
    """Case-insensitive lookup. Bundles are a mix of 'Pilot1_*' and 'pilot18_*',
    which works on Windows but NOT on the (case-sensitive) Pi filesystem."""
    exact = directory / name
    if exact.exists():
        return exact
    low = name.lower()
    for p in directory.iterdir():
        if p.name.lower() == low:
            return p
    raise FileNotFoundError(f"{name} not found in {directory}")


def load_recording(subject: int):
    """Same data the streamer sends: 64 EEG channels, volts, float32 on the wire."""
    from .preprocessing import load_raw          # lazy import: needs MNE
    path = _ci_find(C.DATA_DIR, f"Pilot{subject}.bdf")
    raw = load_raw(path)
    raw.pick_types(eeg=True)
    data = raw.get_data().astype(np.float32).astype(np.float64)   # LSL float32
    return data, list(raw.ch_names), float(raw.info["sfreq"])


# --------------------------------------------------------------------------- #
# Streaming replay
# --------------------------------------------------------------------------- #
def replay_stream(bundle_path, data, ch_names, sfreq_in, chunk: int = 16,
                  apply_mi_bandpass: bool = True):
    """
    Push `data` (n_ch, n_samples @ sfreq_in) through RealTimeMI chunk by chunk
    and return (bci, Z) where Z[:, j] is the j-th 128 Hz sample the online
    path produced (motor channels, fully preprocessed). Decimation starts at
    input sample 0, so Z[:, j] <-> offline resampled sample j.
    """
    bci = RealTimeMI(str(bundle_path), all_ch_names=ch_names,
                     sfreq_in=sfreq_in, apply_mi_bandpass=apply_mi_bandpass)
    parts = []
    n = data.shape[1]
    for s in range(0, n, chunk):
        before = bci.n_produced
        bci.push(data[:, s:s + chunk])
        d = bci.n_produced - before
        if d > 0:
            parts.append(bci._buf[:, -d:].copy())
    return bci, np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _rel_err(W, R):
    return float(np.linalg.norm(W - R) / (np.linalg.norm(R) + 1e-30))


def _chan_corr(W, R):
    return float(np.mean([np.corrcoef(W[c], R[c])[0, 1] for c in range(W.shape[0])]))


def evaluate(b: dict, Z: np.ndarray, win_samples: int, shifts_s, max_align: int = 4):
    """b = joblib bundle (train.py), Z = online 128 Hz stream (see replay_stream)."""
    sf = float(b["sfreq"])
    pipe = b["pipeline"]
    X_ref = np.asarray(b["X_test"])                       # (n, ch, n_ep) offline epochs
    y = np.asarray(b["y_test"])
    n_ep = X_ref.shape[2]
    cue = np.round(np.asarray(b["test_event_times_sec"]) * sf).astype(int)
    start = cue + int(round(b["tmin"] * sf))              # first sample of each epoch

    ok = np.array([(s0 - max_align >= 0) and (s0 + n_ep + max_align <= Z.shape[1])
                   for s0 in start])
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise RuntimeError("no test epoch fits inside the replayed stream")

    # 1) signal level ---------------------------------------------------------
    rel0 = [_rel_err(Z[:, start[i]:start[i] + n_ep], X_ref[i]) for i in idx]
    cor0 = [_chan_corr(Z[:, start[i]:start[i] + n_ep], X_ref[i]) for i in idx]
    align = {}
    for d in range(-max_align, max_align + 1):
        align[d] = float(np.median([_rel_err(Z[:, start[i] + d:start[i] + d + n_ep],
                                             X_ref[i]) for i in idx]))
    best_shift = min(align, key=align.get)

    # 2) prediction level -----------------------------------------------------
    Xon = np.stack([Z[:, start[i]:start[i] + n_ep] for i in idx])
    Xoff = X_ref[idx]
    yy = y[idx]
    preds = {
        "offline_full": pipe.predict(Xoff),
        "online_full": pipe.predict(Xon),
        # what RealTimeMI.try_predict() really feeds the model (WINDOW_LEN samples)
        "offline_win": pipe.predict(Xoff[:, :, -win_samples:]),
        "online_win": pipe.predict(Xon[:, :, -win_samples:]),
    }
    acc = {k: float(accuracy_score(yy, p)) for k, p in preds.items()}
    bacc = {k: float(balanced_accuracy_score(yy, p)) for k, p in preds.items()}
    agree_full = float(np.mean(preds["online_full"] == preds["offline_full"]))
    agree_win = float(np.mean(preds["online_win"] == preds["offline_win"]))

    # 3) alignment sensitivity (online windows ending at cue+TMAX+shift) -------
    end_nom = start + n_ep                                  # exclusive end of nominal epoch
    sens = {}
    for sh in shifts_s:
        d = int(round(sh * sf))
        ends = end_nom + d
        sel = [i for i in range(len(y))
               if ends[i] - win_samples >= 0 and ends[i] <= Z.shape[1]]
        if not sel:
            continue
        Xs = np.stack([Z[:, ends[i] - win_samples:ends[i]] for i in sel])
        p = pipe.predict(Xs)
        sens[float(sh)] = {"n": len(sel),
                           "accuracy": float(accuracy_score(y[sel], p))}

    return {
        "n_test_trials": int(len(y)),
        "n_compared": int(idx.size),
        "epoch_samples": int(n_ep),
        "window_samples": int(win_samples),
        "signal": {
            "median_rel_rms_error": float(np.median(rel0)),
            "p95_rel_rms_error": float(np.percentile(rel0, 95)),
            "median_channel_corr": float(np.median(cor0)),
            "best_alignment_shift_samples": int(best_shift),
            "rel_error_by_shift": {str(k): v for k, v in align.items()},
        },
        "prediction": {
            "accuracy": acc, "balanced_accuracy": bacc,
            "agreement_full_epoch": agree_full,
            "agreement_window": agree_win,
        },
        "alignment_sensitivity": {str(k): v for k, v in sens.items()},
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(tag: str, res: dict):
    s, p = res["signal"], res["prediction"]
    verdict = ("PASS" if s["median_rel_rms_error"] < PASS_REL_ERR
               and p["agreement_window"] >= PASS_AGREEMENT
               and s["best_alignment_shift_samples"] == 0 else "CHECK")
    print(f"\n--- {tag} ---  [{verdict}]")
    print(f"trials compared        : {res['n_compared']}/{res['n_test_trials']} "
          f"(epoch {res['epoch_samples']} samples, window {res['window_samples']})")
    print(f"signal  rel.RMS error  : median {s['median_rel_rms_error']:.4f} | "
          f"p95 {s['p95_rel_rms_error']:.4f}   (verdict < {PASS_REL_ERR})")
    print(f"signal  channel corr   : median {s['median_channel_corr']:.4f}")
    print(f"signal  best alignment : {s['best_alignment_shift_samples']:+d} samples "
          f"(0 expected; +-1 = 8 ms convention offset)")
    print(f"predict agreement      : full epoch {p['agreement_full_epoch']:.3f} | "
          f"window {p['agreement_window']:.3f}   (verdict >= {PASS_AGREEMENT})")
    a = p["accuracy"]
    print(f"accuracy vs y_test     : offline {a['offline_full']:.3f} | "
          f"online {a['online_full']:.3f} (full epoch) | "
          f"offline {a['offline_win']:.3f} | online {a['online_win']:.3f} (window)")
    if res["alignment_sensitivity"]:
        print("accuracy vs window end (relative to cue+TMAX; window length fixed):")
        for sh, r in res["alignment_sensitivity"].items():
            print(f"   {float(sh):+6.3f} s  ->  acc {r['accuracy']:.3f}  (n={r['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--subject", type=int, default=1)
    ap.add_argument("-m", "--model", type=int, choices=range(1, 7), default=3,
                    help="1=svm 2=rf 3=fbcsp 4=mdm 5=tangent_space 6=fb_tangent")
    ap.add_argument("--all-models", action="store_true",
                    help="evaluate all six bundles of the subject on one replay")
    ap.add_argument("--legacy", action="store_true",
                    help="also replay WITHOUT the online 3-30 Hz band-pass "
                         "(the old behaviour) for a before/after comparison")
    ap.add_argument("--chunk", type=int, default=16,
                    help="samples per push, like the streamer (default 16)")
    ap.add_argument("--shifts", type=str,
                    default="-0.5,-0.25,0,0.25,0.5,0.75,1.0",
                    help="window-end shifts in seconds for the alignment table")
    args = ap.parse_args()

    shifts = [float(x) for x in args.shifts.split(",") if x.strip()]
    names = list(C.MODEL_MAP.values()) if args.all_models else [C.MODEL_MAP[args.model]]
    paths = {n: _ci_find(C.OUTPUT_DIR, f"Pilot{args.subject}_{n}.joblib") for n in names}

    print(f"Loading Pilot{args.subject} ...")
    data, ch_names, sfreq_in = load_recording(args.subject)
    print(f"  {data.shape[0]} ch x {data.shape[1]} samples @ {sfreq_in:.0f} Hz")

    variants = [("with online 3-30 Hz band-pass", True)]
    if args.legacy:
        variants.append(("LEGACY: no online 3-30 Hz band-pass", False))

    out = {}
    for vname, apply_mi in variants:
        cache = {}                       # Z depends only on the channel set + filter flag
        for n in names:
            b = joblib.load(paths[n])
            key = tuple(b["channels"])
            if key not in cache:
                print(f"\nReplaying stream ({vname}) ...")
                bci, Z = replay_stream(paths[n], data, ch_names, sfreq_in,
                                       chunk=args.chunk, apply_mi_bandpass=apply_mi)
                cache[key] = (Z, bci.win_samples)
            Z, win = cache[key]
            res = evaluate(b, Z, win, shifts)
            print_report(f"Pilot{args.subject} / {n} / {vname}", res)
            out.setdefault(vname, {})[n] = res

    dest = C.OUTPUT_DIR / "online"
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"Pilot{args.subject}_replay.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nReport written to: {f}")


if __name__ == "__main__":
    main()