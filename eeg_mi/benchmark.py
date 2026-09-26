"""
Standalone inference benchmark for the Raspberry Pi -- no LSL, no streamer.

Purpose: isolate the *computational* cost of each of the six models on the
target board. The receiver measures the whole system (acquisition + filtering +
inference + printing); this script measures only the model, on exactly the same
windows the offline evaluation used (`X_test` stored in each joblib bundle).

That gives the accuracy-vs-cost table that justifies the final model choice:

    model          acc     lat mean   lat p95   RSS    file    ITR
    fbcsp        0.812      18.4 ms   21.0 ms   118MB  2.1MB   36.2 b/min
    ...

Run on the Pi:
    python -m eeg_mi.benchmark -s 1                  # all available models
    python -m eeg_mi.benchmark -s 1 -m 3 -m 6 -n 300 # specific models
    python -m eeg_mi.benchmark -s 1 --all-subjects
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import gc
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, cohen_kappa_score

from . import config as C
from .profiling import (HardwareMonitor, StageTimer, system_info, wolpaw_itr,
                        write_report)


def benchmark_bundle(path: Path, n_windows: int, hw_interval: float) -> dict | None:
    if not path.exists():
        return None

    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss / 2 ** 20

    t0 = time.perf_counter()
    bundle = joblib.load(path)
    load_ms = (time.perf_counter() - t0) * 1e3
    rss_after = proc.memory_info().rss / 2 ** 20

    pipe = bundle["pipeline"]
    X = np.asarray(bundle["X_test"], dtype=np.float64)
    y = np.asarray(bundle["y_test"])
    if X.size == 0:
        return None

    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = np.arange(len(X))
    if n_windows and n_windows < len(X):
        idx = rng.choice(idx, size=n_windows, replace=False)

    # warm-up: first call pays for lazy imports / BLAS pools / page faults
    for _ in range(3):
        pipe.predict(X[:1])

    timer = StageTimer()
    hw = HardwareMonitor(interval=hw_interval, proc=proc)
    hw.start()

    preds = np.empty(len(idx), dtype=int)
    steps = getattr(pipe, "steps", None)
    for k, i in enumerate(idx):
        Xi = X[i:i + 1]
        with timer.track("total"):
            if steps is None:
                preds[k] = int(pipe.predict(Xi)[0])
            else:
                Xt = Xi
                for name, step in steps[:-1]:
                    with timer.track(f"step/{name}"):
                        Xt = step.transform(Xt)
                name, clf = steps[-1]
                with timer.track(f"step/{name}"):
                    preds[k] = int(clf.predict(Xt)[0])

    hw.stop()
    rss_peak = proc.memory_info().rss / 2 ** 20

    stats = timer.summary(reference="total")
    total = stats["total"]
    acc = float(accuracy_score(y[idx], preds))
    dec_per_min = 60.0 / C.WINDOW_STEP        # decision rate the design targets

    return {
        "model": bundle.get("model_name", path.stem.split("_", 1)[-1]),
        "n_windows": int(len(idx)),
        "accuracy": acc,
        "kappa": float(cohen_kappa_score(y[idx], preds)),
        "latency_ms": {k: total[k] for k in
                       ("mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")},
        "stages_ms": stats,
        "footprint": {
            "model_file_mb": path.stat().st_size / 2 ** 20,
            "model_rss_mb": rss_after - rss_before,
            "rss_peak_mb": rss_peak,
            "load_ms": load_ms,
        },
        "hardware": hw.summary(),
        "itr": wolpaw_itr(acc, len(bundle["event_id"]), dec_per_min),
        "_timer": timer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--subject", type=int, default=1)
    ap.add_argument("-m", "--model", type=int, action="append",
                    choices=range(1, 7),
                    help="repeatable; default = every model found")
    ap.add_argument("-n", "--n-windows", type=int, default=200,
                    help="windows to time per model (0 = all test windows)")
    ap.add_argument("--hw-interval", type=float, default=0.25)
    ap.add_argument("--stages", action="store_true",
                    help="print the per-step breakdown for every model")
    args = ap.parse_args()

    models = ([C.MODEL_MAP[i] for i in args.model] if args.model
              else list(C.MODEL_MAP.values()))

    sysinfo = system_info()
    print("=" * 96)
    print(f"Benchmark on {sysinfo['board_model'] or sysinfo['platform']} | "
          f"{sysinfo['cpu_count_logical']} cores | "
          f"{sysinfo['ram_total_mb']:.0f} MB RAM | "
          f"T0 = {sysinfo['temperature_c']} C | "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}")
    print(f"Subject Pilot{args.subject} | window {C.WINDOW_LEN}s @ "
          f"{C.SFREQ_TARGET} Hz | step {C.WINDOW_STEP}s")
    print("=" * 96)

    results = []
    for name in models:
        path = C.OUTPUT_DIR / f"Pilot{args.subject}_{name}.joblib"
        res = benchmark_bundle(path, args.n_windows, args.hw_interval)
        if res is None:
            print(f"  [{name}] bundle not found -- skipped ({path.name})")
            continue
        results.append(res)
        print(f"  [{name}] done ({res['n_windows']} windows)")

    if not results:
        raise SystemExit("No bundles found. Run `python -m eeg_mi.train` first.")

    hdr = (f"\n{'model':>14} {'acc':>7} {'kappa':>7} {'mean':>9} {'p95':>9} "
           f"{'max':>9} {'duty':>7} {'RSS':>8} {'file':>8} {'ITR':>11}")
    print(hdr)
    print(f"{'':>14} {'':>7} {'':>7} {'[ms]':>9} {'[ms]':>9} {'[ms]':>9} "
          f"{'[%]':>7} {'[MB]':>8} {'[MB]':>8} {'[b/min]':>11}")
    print("-" * len(hdr))
    for r in sorted(results, key=lambda r: -r["accuracy"]):
        lat = r["latency_ms"]
        duty = 100.0 * lat["mean_ms"] / (C.WINDOW_STEP * 1e3)
        print(f"{r['model']:>14} {r['accuracy']:>7.3f} {r['kappa']:>7.3f} "
              f"{lat['mean_ms']:>9.2f} {lat['p95_ms']:>9.2f} "
              f"{lat['max_ms']:>9.2f} {duty:>7.1f} "
              f"{r['footprint']['model_rss_mb']:>8.1f} "
              f"{r['footprint']['model_file_mb']:>8.2f} "
              f"{r['itr']['bits_per_min']:>11.1f}")

    if args.stages:
        for r in results:
            r["_timer"].print_table(f"{r['model']} -- per-step cost",
                                    prefix="step/", reference="total")

    payload = {
        "subject": f"Pilot{args.subject}",
        "system": sysinfo,
        "config": {"window_s": C.WINDOW_LEN, "step_s": C.WINDOW_STEP,
                   "sfreq": C.SFREQ_TARGET,
                   "channels": C.MOTOR_CHANNELS_1020,
                   "threads": os.environ.get("OMP_NUM_THREADS")},
        "results": [{k: v for k, v in r.items() if k != "_timer"} for r in results],
    }
    out = C.OUTPUT_DIR / "online" / f"Pilot{args.subject}_benchmark.json"
    write_report(out, payload)
    print(f"\nSaved: {out}\n")


if __name__ == "__main__":
    main()