"""
Online evaluation of a trained model on an LSL stream, instrumented for the
Raspberry Pi deployment chapter of the thesis.

What it measures
----------------
LATENCY (per decision, ms)
    * every preprocessing stage (notch / band-pass / CAR / decimation / buffer)
    * every sklearn pipeline step (e.g. filter-bank CSP, SelectKBest, PCA, SVC)
      with its share (%) of the total compute -> "where does the time go"
    * end-to-end acquisition-to-decision latency, using the LSL timestamps of
      the samples themselves (clock-corrected) -- this is the number that
      matters for a BCI, not the model's compute time alone
    * decision-rate jitter (actual vs. the 0.25 s target step)

THROUGHPUT / FEASIBILITY
    * real-time factor: CPU seconds spent per second of EEG (must stay << 1)
    * duty cycle: compute time per decision / decision period
    * dropped / missing samples vs. the nominal sampling rate

HARDWARE
    * process CPU% (per core and normalised to the whole board), system CPU%
    * RSS and system RAM, sampled at a fixed cadence in a background thread
    * SoC temperature and CPU frequency over time
    * Raspberry Pi throttling / under-voltage flags (vcgencmd get_throttled)

ACCURACY
    * accuracy, balanced accuracy, Cohen's kappa, confusion matrix on the
      test-set windows flagged by the streamer
    * Wolpaw ITR (bits/min), so a slower-but-better model can be compared
      against a faster-but-worse one on a single axis

Everything is printed at the end and written to
`artifacts/online/Pilot<S>_<model>_online.json` (+ an optional per-decision CSV),
so the six models can be compared and plotted.

Run:
    python -m eeg_mi.receiver -s 1 -m 3
    python -m eeg_mi.receiver -s 1 -m 6 --quiet --csv --duration 300
"""

from __future__ import annotations

import os

# BLAS thread pools must be pinned BEFORE numpy is imported. On a 4-core Pi,
# letting OpenBLAS spawn 4 threads for tiny (7 x 128) matrices makes latency
# both higher and far more variable. Override by exporting the vars yourself.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import psutil
from pylsl import StreamInlet, local_clock, resolve_byprop
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score)

from . import config as C
from .online_inference import RealTimeMI
from .profiling import (HardwareMonitor, StageTimer, system_info, wolpaw_itr,
                        write_report)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _connect(stream_name: str | None, timeout: float = 10.0) -> StreamInlet | None:
    print("Searching for an LSL EEG stream...")
    if stream_name:
        streams = resolve_byprop("name", stream_name, timeout=timeout)
    else:
        streams = resolve_byprop("type", "EEG", timeout=timeout)
    if not streams:
        print("No EEG LSL stream was found.")
        return None
    # max_buflen: keep the inlet buffer short. A long buffer hides the fact
    # that the receiver cannot keep up -- samples queue instead of being lost,
    # and the end-to-end latency silently grows.
    inlet = StreamInlet(streams[0], max_buflen=10, max_chunklen=0,
                        recover=False)
    return inlet


def _channel_names(info) -> list[str]:
    names, ch = [], info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        names.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return [n for n in names if n]


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run_receiver(model_path: Path, subject: int, model_name: str,
                 stream_name: str | None = None, warmup_sec: float = 5.0,
                 duration: float | None = None, quiet: bool = False,
                 csv_path: Path | None = None, report_path: Path | None = None,
                 hw_interval: float = 0.5, eos_timeout: float = 5.0) -> dict | None:

    sysinfo = system_info()
    print("=" * 70)
    print(f"Board       : {sysinfo['board_model'] or sysinfo['platform']}")
    print(f"CPU         : {sysinfo['cpu_count_logical']} cores @ "
          f"{_pct(sysinfo['cpu_freq_mhz'])} MHz "
          f"(governor: {sysinfo['cpu_governor']})")
    print(f"RAM         : {sysinfo['ram_total_mb']:.0f} MB  |  "
          f"Temp: {_pct(sysinfo['temperature_c'])} C")
    print(f"Threads     : OMP={os.environ.get('OMP_NUM_THREADS')}")
    print("=" * 70)

    inlet = _connect(stream_name)
    if inlet is None:
        return None

    info = inlet.info()
    sfreq_in = info.nominal_srate()
    ch_names = _channel_names(info)
    eeg_ch_names = ch_names[:-1] if ch_names else None   # drop "GroundTruth"
    print(f"Connected to '{info.name()}' | {info.channel_count()} ch @ "
          f"{sfreq_in} Hz")

    # --- model loading is itself a hardware metric on an embedded board ----
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 2 ** 20
    t0 = time.perf_counter()
    bci = RealTimeMI(str(model_path), all_ch_names=eeg_ch_names,
                     sfreq_in=sfreq_in)
    load_ms = (time.perf_counter() - t0) * 1e3
    rss_after = proc.memory_info().rss / 2 ** 20
    model_mb = Path(model_path).stat().st_size / 2 ** 20

    bci.warm_up(3)          # exclude first-call costs from the measurements

    print(f"Model       : {model_name} | file {model_mb:.2f} MB | "
          f"load {load_ms:.0f} ms | RSS +{rss_after - rss_before:.1f} MB")
    print(f"Inference   : window {bci.win_sec}s, step {C.WINDOW_STEP}s, "
          f"{len(bci.model_channels)} ch @ {bci.sfreq} Hz "
          f"({bci.win_samples} samples/window)")
    print(f"Warm-up     : {warmup_sec:.0f}s (timing counters reset afterwards)\n")

    timer = StageTimer()
    hw = HardwareMonitor(interval=hw_interval, proc=proc)
    hw.start()

    step_sec = C.WINDOW_STEP
    gt_win = int(round(bci.win_sec * sfreq_in))      # GT samples per window
    gt_buf = np.full(gt_win, -1.0)

    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict] = []
    e2e_ms: list[float] = []
    decision_times: list[float] = []
    n_samples_recv = 0
    n_invalid_decisions = 0

    time_corr = inlet.time_correction(timeout=2.0)
    last_corr = time.time()
    last_ts = None

    t_start = time.time()
    last_predict = t_start
    last_data = t_start
    warm_done = warmup_sec <= 0

    label_to_idx = {lbl: i for i, lbl in enumerate(bci.labels)}

    try:
        while True:
            t_pull = time.perf_counter()
            chunk, timestamps = inlet.pull_chunk(timeout=0.2, max_samples=128)
            pull_ms = (time.perf_counter() - t_pull) * 1e3
            now = time.time()

            if chunk:
                last_data = now
                timer.add("io/pull_chunk", pull_ms)
                with timer.track("io/to_numpy"):
                    samples = np.asarray(chunk, dtype=np.float64).T
                n_new = samples.shape[1]
                n_samples_recv += n_new
                last_ts = timestamps[-1]

                # ground-truth ring buffer at the INPUT rate
                gt_buf = np.concatenate([gt_buf, samples[-1, :]])[-gt_win:]

                with timer.track("preprocess/total"):
                    bci.push(samples[:-1, :], timer=timer)
            else:
                if now - last_data > eos_timeout:
                    print(f"\nNo data for {eos_timeout:.0f}s -- stream ended.")
                    break

            if duration and (now - t_start) >= duration:
                print(f"\nReached --duration {duration:.0f}s.")
                break

            if not warm_done and (now - t_start) >= warmup_sec:
                timer.reset()
                hw.reset()
                y_true.clear(); y_pred.clear(); rows.clear()
                e2e_ms.clear(); decision_times.clear()
                n_invalid_decisions = 0
                warm_done = True
                if not quiet:
                    print("--- warm-up finished, counters reset ---")

            # ---------------- decision ----------------
            if (now - last_predict) < step_sec or not bci.ready:
                continue

            with timer.track("decision/total"):
                label, scores = bci.try_predict(timer=timer, return_scores=True)
            t_done = local_clock()

            if label is None:
                continue

            if time.time() - last_corr > 10.0:      # refresh clock offset
                time_corr = inlet.time_correction(timeout=0.5)
                last_corr = time.time()

            interval_ms = (now - last_predict) * 1e3
            last_predict = now
            if decision_times:
                timer.add("decision/interval", interval_ms)
            decision_times.append(now)

            # acquisition -> decision latency (the BCI-relevant one)
            lat_e2e = None
            if last_ts is not None:
                lat_e2e = (t_done - (last_ts + time_corr)) * 1e3
                if -50.0 < lat_e2e < 5000.0:        # ignore clock glitches
                    e2e_ms.append(lat_e2e)
                    timer.add("decision/end_to_end", lat_e2e)

            compute_ms = timer.raw("decision/total")[-1]

            # a window counts only if the WHOLE window sits inside one
            # flagged test-set trial (no straddling of trial boundaries)
            uniq = np.unique(gt_buf)
            valid = uniq.size == 1 and uniq[0] != -1.0
            if valid:
                gt_idx = int(uniq[0])
                gt_str = bci.labels[gt_idx] if gt_idx < len(bci.labels) else str(gt_idx)
                y_true.append(gt_idx)
                y_pred.append(label_to_idx.get(label, -1))
                acc = 100.0 * float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
                if not quiet:
                    hw_now = hw.latest()
                    ok = "TRUE " if label == gt_str else "FALSE"
                    print(f"{label:<9}| gt {gt_str:<9}[{ok}] | "
                          f"acc {acc:>5.1f}% | "
                          f"compute {compute_ms:>5.1f} ms | "
                          f"e2e {('%6.1f' % lat_e2e) if lat_e2e is not None else '   n/a'} ms | "
                          f"cpu {hw_now.get('cpu_proc_pct', float('nan')):>5.1f}% | "
                          f"rss {hw_now.get('rss_mb', float('nan')):>5.1f} MB | "
                          f"T {hw_now.get('temp_c') or float('nan'):>4.1f}C")
            else:
                n_invalid_decisions += 1

            if csv_path is not None:
                rows.append({
                    "t_rel_s": now - t_start,
                    "pred": label,
                    "gt": int(uniq[0]) if valid else -1,
                    "valid": int(valid),
                    "compute_ms": compute_ms,
                    "e2e_ms": lat_e2e,
                    "interval_ms": interval_ms,
                    "scores": None if scores is None else
                              ";".join(f"{v:.4f}" for v in np.atleast_1d(scores)),
                    **{k: hw.latest().get(k) for k in
                       ("cpu_proc_pct", "rss_mb", "temp_c", "freq_mhz")},
                })

    except KeyboardInterrupt:
        print("\nReceiver terminated by user.")
    except Exception as exc:
        print(f"\nStream ended or lost ({exc}). Saving report...")

    elapsed = time.time() - t_start
    hw.stop()

    return _finalise(bci=bci, timer=timer, hw=hw, sysinfo=sysinfo,
                     subject=subject, model_name=model_name,
                     model_path=model_path, model_mb=model_mb,
                     load_ms=load_ms, rss_model_mb=rss_after - rss_before,
                     y_true=y_true, y_pred=y_pred, e2e_ms=e2e_ms,
                     decision_times=decision_times, elapsed=elapsed,
                     sfreq_in=sfreq_in, n_samples_recv=n_samples_recv,
                     n_invalid=n_invalid_decisions, rows=rows,
                     csv_path=csv_path, report_path=report_path)


# --------------------------------------------------------------------------- #
# Final report
# --------------------------------------------------------------------------- #
def _finalise(*, bci, timer, hw, sysinfo, subject, model_name, model_path,
              model_mb, load_ms, rss_model_mb, y_true, y_pred, e2e_ms,
              decision_times, elapsed, sfreq_in, n_samples_recv, n_invalid,
              rows, csv_path, report_path) -> dict:

    n_dec = len(decision_times)
    dec_per_min = 60.0 * n_dec / elapsed if elapsed > 0 else 0.0

    # ---- compute budget ----
    stats = timer.summary(reference="decision/total")
    # "decision/total" already contains the predict/* steps and
    # "preprocess/total" the pre/* steps, so there is no double counting.
    # "io/pull_chunk" is excluded on purpose: it is idle waiting, not CPU work.
    compute_keys = ("decision/total", "preprocess/total", "io/to_numpy")
    compute_sum_ms = sum(stats[k]["sum_ms"] for k in compute_keys if k in stats)
    stream_sec = n_samples_recv / sfreq_in if sfreq_in else elapsed
    rtf = (compute_sum_ms / 1e3) / stream_sec if stream_sec > 0 else None
    mean_decision_ms = stats.get("decision/total", {}).get("mean_ms")
    duty = (mean_decision_ms / (C.WINDOW_STEP * 1e3)) if mean_decision_ms else None

    print("\n" + "=" * 70)
    print(f"  ONLINE REPORT -- Pilot{subject} / {model_name}")
    print("=" * 70)
    print(f"Duration              : {elapsed:.1f} s "
          f"({stream_sec:.1f} s of EEG received)")
    print(f"Decisions             : {n_dec} "
          f"({dec_per_min:.1f}/min, {n_invalid} outside test trials)")
    exp = int(round(elapsed * sfreq_in))
    print(f"Samples received      : {n_samples_recv} / ~{exp} expected "
          f"({100.0 * n_samples_recv / exp if exp else 0:.1f}%)")

    timer.print_table("Per-decision inference (pipeline steps)",
                      prefix="predict/", reference="decision/total")
    timer.print_table("Per-chunk preprocessing",
                      prefix="pre/", reference="preprocess/total")
    timer.print_table("Totals", prefix="decision/",
                      reference="decision/total")
    timer.print_table("LSL I/O", prefix="io/", reference="decision/total")

    print("\n--- Real-time feasibility ---")
    if mean_decision_ms is not None:
        print(f"Compute per decision  : {mean_decision_ms:.1f} ms "
              f"(p95 {stats['decision/total']['p95_ms']:.1f}, "
              f"max {stats['decision/total']['max_ms']:.1f})")
    if duty is not None:
        print(f"Duty cycle            : {100 * duty:.1f}% of the "
              f"{C.WINDOW_STEP * 1e3:.0f} ms decision period")
    if rtf is not None:
        print(f"Real-time factor      : {rtf:.4f} CPU-s per second of EEG "
              f"({'OK' if rtf < 1 else 'TOO SLOW'})")
    if e2e_ms:
        a = np.asarray(e2e_ms)
        print(f"End-to-end latency    : mean {a.mean():.1f} ms | "
              f"p95 {np.percentile(a, 95):.1f} | max {a.max():.1f} "
              f"(sample acquisition -> label)")
    if "decision/interval" in timer:
        iv = np.asarray(timer.raw("decision/interval"))
        print(f"Decision interval     : mean {iv.mean():.1f} ms "
              f"(target {C.WINDOW_STEP * 1e3:.0f}), jitter sd {iv.std():.1f} ms")
    print(f"Model footprint       : {model_mb:.2f} MB on disk, "
          f"+{rss_model_mb:.1f} MB RSS, {load_ms:.0f} ms to load")

    hw.print_summary()

    # ---- accuracy ----
    metrics: dict = {}
    if y_true:
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        metrics = {
            "n_windows": int(yt.size),
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "kappa": float(cohen_kappa_score(yt, yp)),
            "f1_macro": float(f1_score(yt, yp, average="macro")),
            "confusion": confusion_matrix(yt, yp).tolist(),
        }
        itr = wolpaw_itr(metrics["accuracy"], len(bci.labels), dec_per_min)
        metrics["itr"] = itr
        print("\n--- Classification (test-set windows only) ---")
        print(f"Windows               : {metrics['n_windows']}")
        print(f"Accuracy              : {100 * metrics['accuracy']:.2f}%  "
              f"(balanced {100 * metrics['balanced_accuracy']:.2f}%)")
        print(f"Kappa / F1-macro      : {metrics['kappa']:.3f} / "
              f"{metrics['f1_macro']:.3f}")
        print(f"Confusion             : {metrics['confusion']} "
              f"(rows=true {bci.labels})")
        print(f"ITR (Wolpaw)          : {itr['bits_per_decision']:.3f} bits/decision"
              f" -> {itr['bits_per_min']:.1f} bits/min")
    else:
        print("\nNo test-set windows were evaluated "
              "(is the streamer flagging the GroundTruth channel?).")

    payload = {
        "subject": f"Pilot{subject}",
        "model": model_name,
        "model_path": str(model_path),
        "system": sysinfo,
        "run": {
            "elapsed_s": elapsed,
            "eeg_seconds_received": stream_sec,
            "n_decisions": n_dec,
            "decisions_per_min": dec_per_min,
            "n_decisions_outside_test_trials": n_invalid,
            "samples_received": n_samples_recv,
            "samples_expected": exp,
            "window_s": bci.win_sec,
            "step_s": C.WINDOW_STEP,
            "sfreq_in": sfreq_in,
            "sfreq_model": bci.sfreq,
            "channels": bci.model_channels,
            "threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "latency": {
            "stages_ms": stats,
            "end_to_end_ms": {
                "mean": float(np.mean(e2e_ms)) if e2e_ms else None,
                "p95": float(np.percentile(e2e_ms, 95)) if e2e_ms else None,
                "max": float(np.max(e2e_ms)) if e2e_ms else None,
            },
            "duty_cycle": duty,
            "real_time_factor": rtf,
        },
        "footprint": {
            "model_file_mb": model_mb,
            "model_rss_mb": rss_model_mb,
            "model_load_ms": load_ms,
        },
        "hardware": hw.summary(),
        "metrics": metrics,
    }

    if report_path is None:
        report_path = C.OUTPUT_DIR / "online" / f"Pilot{subject}_{model_name}_online.json"
    write_report(report_path, payload)
    print(f"\nReport written to: {report_path}")

    if csv_path is not None and rows:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-decision log : {csv_path}")
    print("=" * 70 + "\n")
    return payload


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--subject", type=int, default=1,
                    help="Subject index (e.g. 1 for Pilot1)")
    ap.add_argument("-m", "--model", type=int, choices=range(1, 7), default=3,
                    help="1=svm 2=rf 3=fbcsp 4=mdm 5=tangent_space 6=fb_tangent")
    ap.add_argument("--stream", type=str, default=None,
                    help="LSL stream name (default: first stream of type EEG)")
    ap.add_argument("--warmup", type=float, default=5.0,
                    help="seconds discarded before timing counters start")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop automatically after N seconds")
    ap.add_argument("--hw-interval", type=float, default=0.5,
                    help="hardware sampling period (s)")
    ap.add_argument("--quiet", action="store_true",
                    help="no per-decision printing (recommended over SSH: "
                         "stdout is slow and inflates the measured latency)")
    ap.add_argument("--csv", action="store_true",
                    help="also write a per-decision CSV for plotting")
    args = ap.parse_args()

    model_name = C.MODEL_MAP[args.model]
    model_file = C.OUTPUT_DIR / f"Pilot{args.subject}_{model_name}.joblib"
    csv_file = (C.OUTPUT_DIR / "online" /
                f"Pilot{args.subject}_{model_name}_decisions.csv") if args.csv else None

    run_receiver(model_file, subject=args.subject, model_name=model_name,
                 stream_name=args.stream, warmup_sec=args.warmup,
                 duration=args.duration, quiet=args.quiet, csv_path=csv_file,
                 hw_interval=args.hw_interval)