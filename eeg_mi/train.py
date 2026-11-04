"""
Train the six classifiers stated in models.py for Movement/Rest motor imagery.

SPLIT STRATEGY
----------------------------------------
EEG is highly subject-specific, and a real-time BCI on a Pi is normally
calibrated per user, resulting in the following setup:

    -> ONE MODEL PER SUBJECT (per .bdf file)
    -> within each subject: stratified train/test hold-out (75/25)
    -> on the training portion: stratified 5-fold CV for model selection
    -> the held-out 25% is the test set the metrics are reported on

We then aggregate the per-subject test metrics (mean +/- std across all
subjects) to characterise the approach as a whole.

Run:
    python -m eeg_mi.train                 # all subjects
    python -m eeg_mi.train --limit x       # quick run on first x files
"""

from __future__ import annotations

import argparse
import json
import warnings
import shutil
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, confusion_matrix,
)

from . import config as C
from .preprocessing import preprocess_file
from .features import select_motor_channels
from .models import build_models

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred) -> dict:
    """Metrics suited to a multi-class, possibly imbalanced BCI problem."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        # Cohen's kappa corrects for chance in BCI.
        "kappa": cohen_kappa_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        # Saving confusion matrices to check for prediction imbalances
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }

def run_subject(path: Path) -> dict | None:
    print(f"\n=== {path.name} ===")
    
    # 1. Loading eeg data, passing them through the preprocess pipeline and selecting channels
    epochs = preprocess_file(path)
    epochs = select_motor_channels(epochs)
    
    # 2. Extract mi trials
    raw_X = epochs.get_data(copy=True)
    
    # Mapping raw triggers to numbered classes
    code_to_idx = {}
    for idx, (class_name, codes) in enumerate(C.EVENT_ID.items()):
        if isinstance(codes, list):
            for code in codes:
                code_to_idx[code] = idx
        else:
            code_to_idx[codes] = idx
    raw_y = np.array([code_to_idx[code] for code in epochs.events[:, 2]])

    print(f"  {len(raw_y)} raw trials | class counts: "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(raw_y, return_counts=True))} }")

    if len(np.unique(raw_y)) < 2 or len(raw_y) < 4 * C.CV_FOLDS:
        print("  not enough trials/classes -> skipped")
        return None

    # 3. Splitting (75% Train / 25% Test)
    idx_all = np.arange(len(raw_y)) # Trial index to isolate events for online process
    X_tr_raw, X_te_raw, y_tr_raw, y_te_raw, idx_tr, idx_te = train_test_split(
        raw_X, raw_y, idx_all, test_size=C.TEST_SIZE, stratify=raw_y,
        random_state=C.RANDOM_STATE
    )

    test_events = epochs.events[idx_te]      # (n_test_trials, 3): [sample, 0, code]
    train_events = epochs.events[idx_tr]

    test_times_sec = (test_events[:, 0] / epochs.info["sfreq"]).tolist()
    train_times_sec = (train_events[:, 0] / epochs.info["sfreq"]).tolist()
    
    X_tr, y_tr = X_tr_raw, y_tr_raw
    X_te, y_te = X_te_raw, y_te_raw

    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    print(f"  Train windows: {len(y_tr)} | Test windows: {len(y_te)}")

    results, fitted = {}, {}
    for name, model in build_models().items():
        try:
            cv_acc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
            model.fit(X_tr, y_tr)
            test_metrics = evaluate(y_te, model.predict(X_te))
        except Exception as exc:
            print(f"  [{name}] failed: {exc}")
            continue
            
        test_metrics["cv_accuracy"] = float(cv_acc)
        results[name] = test_metrics
        fitted[name] = model
        print(f"  {name:>12} | cv_acc={cv_acc:.3f} "
              f"test_acc={test_metrics['accuracy']:.3f} "
              f"kappa={test_metrics['kappa']:.3f} "
              f"f1={test_metrics['f1_macro']:.3f}")

    if not results:
        return None

    # Determining best model (default best is according to cv accuracy) and saving all trained models
    best = max(results, key=lambda n: results[n]["cv_accuracy"])
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, trained_pipeline in fitted.items():
            bundle = {
                "pipeline": trained_pipeline,
                "model_name": model_name,
                "channels": epochs.ch_names,
                "sfreq": C.SFREQ_TARGET,
                "bandpass": C.BROAD_BANDPASS,  # Το σωστό φίλτρο που διόρθωσες
                "event_id": C.EVENT_ID,
                "tmin": C.TMIN, "tmax": C.TMAX,
                "X_test": X_te,
                "y_test": y_te,
                "test_event_times_sec": test_times_sec,
                "test_event_triggers": test_events[:, 2].tolist(),
                "train_event_times_sec": train_times_sec,
                "train_event_triggers": train_events[:, 2].tolist(),
            }
            # Creating the output files e.g. Pilot1_svm.joblib, Pilot1_fbcsp.joblib ...
            joblib.dump(bundle, C.OUTPUT_DIR / f"{path.stem}_{model_name}.joblib")

    
    print(f"  -> saved all models, best was '{best}'")

    return {"subject": path.stem, "best": best, "results": results}

# --------------------------------------------------------------------------- #
# Aggregation across subjects
# --------------------------------------------------------------------------- #
def summarise(all_results: list[dict]) -> dict:
    summary = {}
    metric_keys = ["accuracy", "balanced_accuracy", "kappa", "f1_macro"]
    model_names = sorted({m for r in all_results for m in r["results"]})
    for name in model_names:
        vals = {k: [] for k in metric_keys}
        for r in all_results:
            if name in r["results"]:
                for k in metric_keys:
                    vals[k].append(r["results"][name][k])
        summary[name] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for k, v in vals.items()
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N files (quick test)")
    ap.add_argument("--specific", type=int, default=None,
                    help="process only the Nth file (1-based index)")
    args = ap.parse_args()

    # # Removing outdated directory to ensure clean output data
    # if C.OUTPUT_DIR.exists():
    #     shutil.rmtree(C.OUTPUT_DIR)

    files = sorted(C.DATA_DIR.glob(C.FILE_GLOB))
    if args.limit:
        files = files[: args.limit]
    if args.specific:
        files = [files[args.specific - 1]]
    if not files:
        raise SystemExit(f"No files matching {C.FILE_GLOB} in {C.DATA_DIR}")

    all_results = []
    for path in files:
        try:
            res = run_subject(path)
            if res:
                all_results.append(res)
        except Exception as exc:
            print(f"  ERROR on {path.name}: {exc}")

    summary = summarise(all_results)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"per_subject": all_results, "across_subjects": summary},
                   indent=2))

    print("\n================  ACROSS-SUBJECT SUMMARY  ================")
    print(f"{'model':>5} | {'acc':>13} | {'bal_acc':>13} | "
          f"{'kappa':>13} | {'f1_macro':>13}")
    for name, s in summary.items():
        print(f"{name:>5} | "
              f"{s['accuracy']['mean']:.3f}+/-{s['accuracy']['std']:.3f} | "
              f"{s['balanced_accuracy']['mean']:.3f}+/-{s['balanced_accuracy']['std']:.3f} | "
              f"{s['kappa']['mean']:.3f}+/-{s['kappa']['std']:.3f} | "
              f"{s['f1_macro']['mean']:.3f}+/-{s['f1_macro']['std']:.3f}")
    print(f"\nArtifacts + summary.json written to: {C.OUTPUT_DIR}")


if __name__ == "__main__":
    main()