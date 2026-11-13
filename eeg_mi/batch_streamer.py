"""
LAPTOP side of the automated Pi runs (the MASTER: it owns the job list).

For every (subject, model) job it starts an LSL stream called
    eegsim_s<subject>_m<model>_a<attempt>
and WAITS until a receiver connects before sending a single sample. The Pi
side (batch_receiver.py) is a stateless worker that runs whatever job it
finds advertised, so no clock/ssh synchronisation is needed: start the Pi
worker once (it can wait for hours), then start this script.

If the receiver vanishes mid-stream (Pi crashed / power-cycled / rebooted by
its watchdog) the stream is aborted and the SAME job is offered again under a
new attempt number; the Pi worker, once it is back, picks it up and
overwrites that job's report.

Examples (from the folder that contains the eeg_mi package):
    python -m eeg_mi.batch_streamer --subjects 28,1,31 --models 3,6 --max-sec 300
    python -m eeg_mi.batch_streamer --jobs 28:3,28:6,1:3 --max-sec 300
    python -m eeg_mi.batch_streamer --subjects 28,1,31 --models 3 --dry-run
Model ids: 1=svm 2=rf 3=fbcsp 4=mdm 5=tangent_space 6=fb_tangent
"""

from __future__ import annotations

import argparse
import datetime
import time

from . import config as C
from .lsl_streamer import start_stream, _ci_find


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _log(path, msg: str):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="1", help="e.g. 28,1,31 (priority order)")
    ap.add_argument("--models", type=str, default="3", help="e.g. 3,6")
    ap.add_argument("--jobs", type=str, default=None,
                    help="explicit list 'subject:model,...' (overrides --subjects/--models)")
    ap.add_argument("--max-sec", type=float, default=None,
                    help="stream only the first N seconds of each recording "
                         "(enough for load profiling; default: whole file)")
    ap.add_argument("--wait-consumer", type=float, default=900.0,
                    help="seconds to wait for the Pi worker per attempt "
                         "(must cover a reboot + boot-time service start)")
    ap.add_argument("--abort-after", type=float, default=30.0,
                    help="abort a stream when no receiver is connected for N seconds")
    ap.add_argument("--retries", type=int, default=2, help="extra attempts per job")
    ap.add_argument("--gap", type=float, default=5.0, help="pause between jobs (s)")
    ap.add_argument("--start-at", type=int, default=0, help="skip the first N jobs (resume)")
    ap.add_argument("--dry-run", action="store_true", help="only print the job list")
    args = ap.parse_args()

    if args.jobs:
        jobs = [tuple(int(v) for v in j.split(":")) for j in args.jobs.split(",") if j.strip()]
    else:
        jobs = [(s, m) for s in _ints(args.subjects) for m in _ints(args.models)]
    log = C.OUTPUT_DIR / "online" / "batch_streamer_log.txt"

    est = (args.max_sec or 33 * 60) / 60.0
    print(f"{len(jobs)} jobs, ~{est:.1f} min of streaming each "
          f"(+ load/cool-down), ~{len(jobs) * (est + 2) / 60:.1f} h in total")
    for i, (s, m) in enumerate(jobs):
        print(f"  {i:>3}: Pilot{s} / {C.MODEL_MAP[m]}")
    if args.dry_run:
        return

    for idx, (s, m) in enumerate(jobs):
        if idx < args.start_at:
            continue
        model_name = C.MODEL_MAP[m]
        bdf = _ci_find(C.DATA_DIR, f"Pilot{s}.bdf")
        bundle = _ci_find(C.OUTPUT_DIR, f"Pilot{s}_{model_name}.joblib")
        status = "?"
        for attempt in range(1, args.retries + 2):
            name = f"eegsim_s{s}_m{m}_a{attempt}"
            _log(log, f"job {idx}/{len(jobs)} Pilot{s}/{model_name} attempt {attempt}: start '{name}'")
            status = start_stream(bdf, bundle_path=bundle, stream_name=name,
                                  wait_consumer=args.wait_consumer, max_sec=args.max_sec,
                                  abort_on_consumer_loss=args.abort_after,
                                  source_id=f"{name}_{int(time.time())}")
            _log(log, f"job {idx} Pilot{s}/{model_name} attempt {attempt}: {status}")
            if status in ("done", "interrupted"):
                break
        if status == "interrupted":
            _log(log, f"interrupted -- resume with --start-at {idx}")
            break
        if status != "done":
            _log(log, f"job {idx} Pilot{s}/{model_name} FAILED after {args.retries + 1} attempts")
        time.sleep(args.gap)
    _log(log, "batch finished")


if __name__ == "__main__":
    main()