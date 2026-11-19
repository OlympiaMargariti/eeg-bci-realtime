"""Benchmark παραλληλοποίησης του filter bank στο Raspberry Pi.

Μετρά τον χρόνο μετασχηματισμού ΕΝΟΣ παραθύρου (όπως στην online λειτουργία) με:
  pipeline      ολόκληρο το pipeline (predict) -- σημείο αναφοράς
  serial        το filter bank όπως είναι σήμερα
  serial+sos    ίδιο, με προϋπολογισμένους συντελεστές φίλτρων (οι συντελεστές
                σήμερα υπολογίζονται ξανά σε κάθε παράθυρο, για κάθε ζώνη)
  threads-N     οι ζώνες μοιράζονται σε N νήματα (κοινή μνήμη, καμία αντιγραφή)
  processes-N   οι ζώνες μοιράζονται σε N διεργασίες (προαιρετικό, --processes)

Η έξοδος κάθε παραλλαγής συγκρίνεται αριθμητικά με τη σειριακή: η παραλληλοποίηση
δεν επιτρέπεται να αλλάξει ούτε ένα χαρακτηριστικό.

Δεν χρειάζεται LSL ή streamer: τρέχει μόνο του στο Pi.

Χρήση (μέσα στον φάκελο που περιέχει το πακέτο eeg_mi):
    python -m eeg_mi.bench_parallel                 # Pilot1, fbcsp, 200 επαναλήψεις
    python -m eeg_mi.bench_parallel -s 4 --reps 400
    python -m eeg_mi.bench_parallel --processes     # και με διεργασίες
"""
from __future__ import annotations

import os
# ίδιες συνθήκες με τον receiver: μία νηματική εκτέλεση ανά βιβλιοθήκη BLAS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, sosfilt

from . import config as C

_WORKER = {}


# --------------------------------------------------------------------------- #
# Βοηθητικά
# --------------------------------------------------------------------------- #
def _ci_find(directory: Path, name: str) -> Path:
    exact = directory / name
    if exact.exists():
        return exact
    low = name.lower()
    for p in directory.iterdir():
        if p.name.lower() == low:
            return p
    raise FileNotFoundError(f"{name} not found in {directory}")


def _read(path, cast=str, default=None):
    try:
        return cast(Path(path).read_text().strip())
    except Exception:
        return default


def hardware():
    t = _read("/sys/class/thermal/thermal_zone0/temp", float)
    return {
        "cores": os.cpu_count(),
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", str, "?"),
        "freq_mhz": (_read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", float) or 0) / 1000.0,
        "temp_c": (t / 1000.0) if t else None,
    }


def timeit(fn, reps, warmup=20):
    for _ in range(warmup):
        fn()
    t = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter()
        fn()
        t[i] = time.perf_counter() - t0
    return {"median_ms": float(np.median(t) * 1e3), "mean_ms": float(t.mean() * 1e3),
            "p95_ms": float(np.percentile(t, 95) * 1e3), "min_ms": float(t.min() * 1e3)}


# --------------------------------------------------------------------------- #
# Παραλλαγές του filter bank
# --------------------------------------------------------------------------- #
def make_variants(fb, X):
    """fb = το βήμα 'fb_csp' του pipeline (FilterBankCSP). Επιστρέφει {όνομα: callable}."""
    bands, csps = list(fb.bands), list(fb.csps)
    sfreq = float(getattr(fb, "sfreq", C.SFREQ_TARGET))
    sos = [butter(4, [lo, hi], btype="band", fs=sfreq, output="sos") for lo, hi in bands]

    def serial():
        return fb.transform(X)

    def band(i):
        return csps[i].transform(sosfilt(sos[i], X, axis=-1))

    def serial_sos():
        return np.concatenate([band(i) for i in range(len(bands))], axis=1)

    def threaded(ex):
        def run():
            return np.concatenate(list(ex.map(band, range(len(bands)))), axis=1)
        return run

    return serial, serial_sos, threaded, len(bands)


def _init_worker(bundle_path, step):
    b = joblib.load(bundle_path)
    fb = b["pipeline"].named_steps[step]
    sfreq = float(getattr(fb, "sfreq", 128))
    _WORKER["csps"] = list(fb.csps)
    _WORKER["sos"] = [butter(4, [lo, hi], btype="band", fs=sfreq, output="sos") for lo, hi in fb.bands]


def _proc_band(args):
    i, X = args
    return _WORKER["csps"][i].transform(sosfilt(_WORKER["sos"][i], X, axis=-1))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--subject", type=int, default=1)
    ap.add_argument("-m", "--model", type=str, default="fbcsp",
                    help="fbcsp (προεπιλογή) ή fb_tangent")
    ap.add_argument("--reps", type=int, default=200, help="χρονομετρημένες επαναλήψεις ανά παραλλαγή")
    ap.add_argument("--threads", type=int, nargs="*", default=[2, 3, 4])
    ap.add_argument("--processes", action="store_true", help="να μετρηθούν και διεργασίες (αργεί)")
    ap.add_argument("--out", default=None, help="αρχείο JSON (προεπιλογή: artifacts/online/bench_parallel.json)")
    a = ap.parse_args()

    step = "fb_csp" if a.model == "fbcsp" else "fb_tangent"
    path = _ci_find(C.OUTPUT_DIR, f"Pilot{a.subject}_{a.model}.joblib")
    bundle = joblib.load(path)
    pipe = bundle["pipeline"]
    if step not in pipe.named_steps:
        raise SystemExit(f"Το μοντέλο '{a.model}' δεν έχει βήμα filter bank ({step}).")
    fb = pipe.named_steps[step]

    X = np.asarray(bundle["X_test"])[:1]          # ένα παράθυρο, όπως στην online λειτουργία
    hw0 = hardware()
    print("=" * 70)
    t0 = f" | T0 {hw0['temp_c']:.1f} C" if hw0["temp_c"] is not None else ""
    print(f"Benchmark παραλληλοποίησης | {platform.machine()} | {hw0['cores']} πυρήνες | "
          f"governor {hw0['governor']}{t0}")
    print(f"Μοντέλο: Pilot{a.subject}/{a.model} | παράθυρο {X.shape[1]} κανάλια x {X.shape[2]} δείγματα | "
          f"{len(fb.bands)} ζώνες | OMP={os.environ['OMP_NUM_THREADS']} | {a.reps} επαναλήψεις")
    print("=" * 70)

    serial, serial_sos, threaded, n_bands = make_variants(fb, X)
    ref = serial()

    results = {}
    def run(name, fn):
        out = fn()
        d = timeit(fn, a.reps)
        d["max_abs_diff"] = float(np.max(np.abs(np.asarray(out) - ref)))
        results[name] = d
        sp = results["serial"]["median_ms"] / d["median_ms"] if "serial" in results else 1.0
        print(f"  {name:<14} {d['median_ms']:7.2f} ms  (p95 {d['p95_ms']:7.2f})  "
              f"x{sp:4.2f}   max|Δ| {d['max_abs_diff']:.2e}")

    print("\nfilter bank (ένα παράθυρο):")
    run("serial", serial)
    run("serial+sos", serial_sos)
    pools = []
    for n in a.threads:
        ex = ThreadPoolExecutor(max_workers=n); pools.append(ex)
        run(f"threads-{n}", threaded(ex))

    if a.processes:
        from multiprocessing import Pool
        n = max(a.threads) if a.threads else 4
        with Pool(n, initializer=_init_worker, initargs=(str(path), step)) as pool:
            def proc():
                return np.concatenate(pool.map(_proc_band, [(i, X) for i in range(n_bands)]), axis=1)
            run(f"processes-{n}", proc)

    print("\nολόκληρο το pipeline (για αναφορά):")
    d = timeit(lambda: pipe.predict(X), a.reps)
    results["pipeline_predict"] = d
    print(f"  {'predict':<14} {d['median_ms']:7.2f} ms  (p95 {d['p95_ms']:7.2f})")

    for ex in pools:
        ex.shutdown()
    hw1 = hardware()
    best = min((k for k in results if k.startswith("threads")),
               key=lambda k: results[k]["median_ms"], default=None)
    if best:
        g = results["serial"]["median_ms"] - results[best]["median_ms"]
        print(f"\nΚαλύτερη παραλλαγή: {best} -> κέρδος {g:.1f} ms ανά απόφαση "
              f"({100 * g / results['serial']['median_ms']:.0f} % του filter bank)")
        print(f"Εκτιμώμενος χρόνος απόφασης: {results['pipeline_predict']['median_ms'] - g:.1f} ms "
              f"από {results['pipeline_predict']['median_ms']:.1f} ms")
    if hw0["temp_c"] is not None:
        print(f"T0 {hw0['temp_c']:.1f} C -> T1 {hw1['temp_c']:.1f} C | "
              f"freq {hw0['freq_mhz']:.0f} -> {hw1['freq_mhz']:.0f} MHz")

    out = Path(a.out) if a.out else (C.OUTPUT_DIR / "online" / "bench_parallel.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"subject": a.subject, "model": a.model, "reps": a.reps,
                               "n_bands": n_bands, "window": list(X.shape),
                               "hw_start": hw0, "hw_end": hw1, "results": results}, indent=2))
    print(f"\nΑποτελέσματα: {out}")


if __name__ == "__main__":
    main()
