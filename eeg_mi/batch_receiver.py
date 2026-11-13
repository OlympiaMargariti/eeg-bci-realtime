"""
Raspberry Pi side of the automated runs: a stateless WORKER.

It polls LSL for streams called  eegsim_s<subject>_m<model>_a<attempt>
(advertised by batch_streamer.py on the laptop), runs the instrumented
receiver on each one as soon as it appears, writes the usual
artifacts/online/Pilot<S>_<model>_online.json, lets the SoC cool down, and
waits for the next stream. The laptop owns the job list and the retries, so
this process can be (re)started at any time -- including automatically at
boot after a power cycle (see the systemd unit in the instructions).

    python -m eeg_mi.batch_receiver                  # run forever
    python -m eeg_mi.batch_receiver --once           # one job, then exit
    python -m eeg_mi.batch_receiver --cool-to 50     # wait until SoC <= 50 C between jobs
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import time
import traceback

from pylsl import resolve_byprop

from . import config as C
from .lsl_receiver import run_receiver, _ci_find
from .profiling import system_info

PATTERN = re.compile(r"^eegsim_s(\d+)_m(\d+)_a(\d+)$")


def _log(path, msg: str):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass


def _cool_down(target_c: float | None, timeout: float, log) -> None:
    """Wait for the SoC to drop below target_c so consecutive jobs start from a
    comparable thermal state (otherwise later jobs inherit the previous heat)."""
    if not target_c:
        return
    t0 = time.time()
    while True:
        temp = system_info().get("temperature_c")
        if temp is None or temp <= target_c:
            return
        if time.time() - t0 > timeout:
            _log(log, f"cool-down timeout, starting at {temp:.1f} C")
            return
        time.sleep(5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-poll", type=float, default=5.0, help="LSL poll window (s)")
    ap.add_argument("--cool-to", type=float, default=None,
                    help="wait until the SoC is at most this many C before each job")
    ap.add_argument("--cool-timeout", type=float, default=300.0)
    ap.add_argument("--csv", action="store_true", help="also write per-decision CSVs")
    ap.add_argument("--once", action="store_true", help="exit after one job")
    args = ap.parse_args()

    log = C.OUTPUT_DIR / "online" / "batch_receiver_log.txt"
    _log(log, "worker started, waiting for eegsim_* streams")
    seen: set[str] = set()

    while True:
        _cool_down(args.cool_to, args.cool_timeout, log)
        streams = resolve_byprop("type", "EEG", timeout=args.prefix_poll)
        cand = [i for i in streams if PATTERN.match(i.name()) and i.name() not in seen]
        if not cand:
            time.sleep(2.0)
            continue

        name = cand[0].name()
        seen.add(name)
        s, m, a = (int(x) for x in PATTERN.match(name).groups())
        model_name = C.MODEL_MAP[m]
        _log(log, f"job Pilot{s}/{model_name} attempt {a}: connecting to '{name}'")
        t0 = time.time()
        try:
            model_file = _ci_find(C.OUTPUT_DIR, f"Pilot{s}_{model_name}.joblib")
            csv_path = (C.OUTPUT_DIR / "online" / f"Pilot{s}_{model_name}_decisions.csv"
                        if args.csv else None)
            res = run_receiver(model_file, subject=s, model_name=model_name,
                               stream_name=name, quiet=True, csv_path=csv_path,
                               wait_stream=30.0)
            if res is None:
                _log(log, f"job Pilot{s}/{model_name}: stream vanished before connecting")
            else:
                met = res.get("metrics", {})
                _log(log, f"job Pilot{s}/{model_name} done in {time.time() - t0:.0f}s | "
                          f"trials {met.get('n_trials')} acc {met.get('accuracy')} | "
                          f"rtf {res['latency'].get('real_time_factor')}")
        except Exception:
            _log(log, f"job Pilot{s}/{model_name} ERROR:\n{traceback.format_exc()}")
        if hasattr(os, "sync"):
            os.sync()                      # flush the report to the SD card before anything else happens
        if args.once:
            break


if __name__ == "__main__":
    main()