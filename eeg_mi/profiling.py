"""
Hardware / latency profiling utilities for the online (Raspberry Pi) BCI.

Provides:
  * StageTimer      -- accumulates per-stage wall-clock times (ms) and reports
                       mean / std / p50 / p95 / p99 / max plus the share (%) of
                       each stage relative to a reference stage.
  * HardwareMonitor -- background thread sampling process CPU%, system CPU%,
                       RSS, SoC temperature, CPU frequency and (on the Pi)
                       the `vcgencmd get_throttled` flags.
  * system_info()   -- one-shot snapshot of the platform (board model, cores,
                       RAM, governor, library versions) for the thesis setup
                       table.
  * wolpaw_itr()    -- Information Transfer Rate (bits/min), the standard BCI
                       throughput metric.

Everything degrades gracefully on a laptop (no vcgencmd / no thermal zone),
so the exact same receiver script runs on both machines.
"""

from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import psutil


# --------------------------------------------------------------------------- #
# 1. Stage timing
# --------------------------------------------------------------------------- #
class StageTimer:
    """
    Accumulates durations per named stage.

    Usage:
        timer = StageTimer()
        with timer.track("predict/fb_csp"):
            ...
        timer.add("io/pull_chunk", ms)      # if already measured
        timer.reset()                       # drop warm-up measurements
    """

    def __init__(self) -> None:
        self._ms: dict[str, list[float]] = defaultdict(list)

    # --- recording ---------------------------------------------------------
    @contextmanager
    def track(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._ms[name].append((time.perf_counter() - t0) * 1e3)

    def add(self, name: str, ms: float) -> None:
        self._ms[name].append(float(ms))

    def reset(self) -> None:
        """Forget everything recorded so far (used after the warm-up period)."""
        self._ms.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._ms

    def raw(self, name: str) -> list[float]:
        return list(self._ms.get(name, []))

    # --- reporting ---------------------------------------------------------
    def summary(self, reference: str | None = None) -> dict:
        """
        {stage: {n, mean_ms, std_ms, p50_ms, p95_ms, p99_ms, max_ms, sum_ms,
                 share_pct}}

        `share_pct` = sum(stage) / sum(reference) * 100. Summing (instead of
        comparing means) is what you want here, because per-chunk stages are
        called far more often than per-decision stages.
        """
        ref_sum = float(np.sum(self._ms[reference])) if reference in self._ms else None
        out: dict[str, dict] = {}
        for name, values in sorted(self._ms.items()):
            a = np.asarray(values, dtype=float)
            if a.size == 0:
                continue
            out[name] = {
                "n": int(a.size),
                "mean_ms": float(a.mean()),
                "std_ms": float(a.std()),
                "p50_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)),
                "p99_ms": float(np.percentile(a, 99)),
                "max_ms": float(a.max()),
                "sum_ms": float(a.sum()),
            }
            if ref_sum:
                out[name]["share_pct"] = 100.0 * float(a.sum()) / ref_sum
        return out

    def print_table(self, title: str, prefix: str = "", reference: str | None = None,
                    ndigits: int = 2) -> None:
        stats = self.summary(reference=reference)
        rows = {k: v for k, v in stats.items() if k.startswith(prefix)}
        if not rows:
            return
        width = max(len(k) for k in rows) + 1
        print(f"\n--- {title} ---")
        print(f"{'stage':<{width}}{'n':>7}{'mean':>9}{'p50':>9}{'p95':>9}"
              f"{'p99':>9}{'max':>9}{'share':>9}")
        print(f"{'':<{width}}{'':>7}{'[ms]':>9}{'[ms]':>9}{'[ms]':>9}"
              f"{'[ms]':>9}{'[ms]':>9}{'[%]':>9}")
        for name, s in sorted(rows.items(), key=lambda kv: -kv[1]["sum_ms"]):
            share = f"{s['share_pct']:.1f}" if "share_pct" in s else "-"
            print(f"{name:<{width}}{s['n']:>7}"
                  f"{s['mean_ms']:>9.{ndigits}f}{s['p50_ms']:>9.{ndigits}f}"
                  f"{s['p95_ms']:>9.{ndigits}f}{s['p99_ms']:>9.{ndigits}f}"
                  f"{s['max_ms']:>9.{ndigits}f}{share:>9}")


# --------------------------------------------------------------------------- #
# 2. Low-level hardware probes
# --------------------------------------------------------------------------- #
_THERMAL_ZONES = sorted(Path("/sys/class/thermal").glob("thermal_zone*")) \
    if Path("/sys/class/thermal").exists() else []


def read_temperature_c() -> float | None:
    """SoC temperature in degrees C (Raspberry Pi: thermal_zone0 = cpu-thermal)."""
    for zone in _THERMAL_ZONES:
        try:
            value = int((zone / "temp").read_text().strip())
            if value > 1000:          # milli-degrees
                value /= 1000.0
            if 0 < value < 150:
                return float(value)
        except Exception:
            continue
    try:                               # laptops (lm-sensors / coretemp)
        temps = psutil.sensors_temperatures()
        for entries in temps.values():
            for entry in entries:
                if entry.current and 0 < entry.current < 150:
                    return float(entry.current)
    except Exception:
        pass
    return None


def read_cpu_freq_mhz() -> float | None:
    try:
        freq = psutil.cpu_freq()
        if freq and freq.current:
            return float(freq.current)
    except Exception:
        pass
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if path.exists():
        try:
            return int(path.read_text().strip()) / 1000.0   # kHz -> MHz
        except Exception:
            return None
    return None


_THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "arm_freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_occurred",
    17: "arm_freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


def read_throttled() -> dict | None:
    """
    Decode `vcgencmd get_throttled` (Raspberry Pi only).

    IMPORTANT for the thesis: if `under_voltage_occurred` or
    `throttled_occurred` is True, every latency number you report was taken on
    a down-clocked board and must be re-measured with a proper PSU / cooling.
    """
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=2.0)
        if out.returncode != 0:
            return None
        raw = out.stdout.strip().split("=")[-1]
        value = int(raw, 16)
    except Exception:
        return None
    flags = {name: bool(value & (1 << bit)) for bit, name in _THROTTLE_BITS.items()}
    flags["raw"] = hex(value)
    return flags


def _read_first_line(path: str) -> str | None:
    try:
        return Path(path).read_text(errors="ignore").strip("\x00").strip()
    except Exception:
        return None


def system_info() -> dict:
    """One-shot platform snapshot -- paste straight into the thesis setup table."""
    info = {
        "board_model": _read_first_line("/proc/device-tree/model"),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_governor": _read_first_line(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_freq_mhz": read_cpu_freq_mhz(),
        "ram_total_mb": psutil.virtual_memory().total / 2 ** 20,
        "swap_total_mb": psutil.swap_memory().total / 2 ** 20,
        "temperature_c": read_temperature_c(),
        "throttled": read_throttled(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for mod in ("numpy", "scipy", "sklearn", "mne", "pyriemann", "joblib", "pylsl"):
        try:
            info[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            info[f"{mod}_version"] = None
    return info


# --------------------------------------------------------------------------- #
# 3. Background hardware monitor
# --------------------------------------------------------------------------- #
class HardwareMonitor(threading.Thread):
    """
    Samples CPU / RAM / temperature in a background thread.

    Why a thread: `psutil.Process.cpu_percent()` is only meaningful over a
    fixed interval. Calling it inside the inference loop (as in the original
    receiver) returns 0.0 on the first call and a value over an ill-defined
    window afterwards. Sampling at a fixed 0.5 s cadence gives a defensible
    CPU-load time series that you can also plot against the temperature curve.
    """

    def __init__(self, interval: float = 0.5, proc: psutil.Process | None = None,
                 throttle_every: int = 10) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.proc = proc or psutil.Process()
        self.throttle_every = throttle_every
        self.samples: list[dict] = []
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._n_cores = psutil.cpu_count(logical=True) or 1
        # prime the CPU counters so the first real sample is not 0.0
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        self.throttle_start = read_throttled()
        self.throttle_flags: dict[str, bool] = {}

    def run(self) -> None:
        i = 0
        while not self._stop_evt.wait(self.interval):
            sample = {
                "t": time.time(),
                "cpu_proc_pct": self.proc.cpu_percent(None),
                "cpu_sys_pct": psutil.cpu_percent(None),
                "rss_mb": self.proc.memory_info().rss / 2 ** 20,
                "mem_used_pct": psutil.virtual_memory().percent,
                "temp_c": read_temperature_c(),
                "freq_mhz": read_cpu_freq_mhz(),
            }
            if self.throttle_every and i % self.throttle_every == 0:
                flags = read_throttled()
                if flags:
                    for key, val in flags.items():
                        if isinstance(val, bool) and val:
                            self.throttle_flags[key] = True
            with self._lock:
                self.samples.append(sample)
            i += 1

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self.join(timeout=2 * self.interval + 1.0)
        except RuntimeError:
            pass

    def reset(self) -> None:
        """Drop warm-up samples."""
        with self._lock:
            self.samples.clear()

    def latest(self) -> dict:
        with self._lock:
            return dict(self.samples[-1]) if self.samples else {}

    def summary(self) -> dict:
        with self._lock:
            samples = list(self.samples)
        if not samples:
            return {"n_samples": 0}

        def agg(key: str) -> dict | None:
            vals = [s[key] for s in samples if s.get(key) is not None]
            if not vals:
                return None
            a = np.asarray(vals, dtype=float)
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "p95": float(np.percentile(a, 95)),
                    "min": float(a.min()), "max": float(a.max())}

        out = {
            "n_samples": len(samples),
            "interval_s": self.interval,
            "n_cores": self._n_cores,
            "cpu_proc_pct": agg("cpu_proc_pct"),
            "cpu_sys_pct": agg("cpu_sys_pct"),
            "rss_mb": agg("rss_mb"),
            "mem_used_pct": agg("mem_used_pct"),
            "temp_c": agg("temp_c"),
            "freq_mhz": agg("freq_mhz"),
            "throttle_at_start": self.throttle_start,
            "throttle_flags_seen": self.throttle_flags or None,
            "throttle_at_end": read_throttled(),
        }
        # psutil reports process CPU as a fraction of ONE core, so on a 4-core
        # Pi a fully busy single-threaded process shows ~100% while the board
        # is only 25% loaded. Report both to avoid the classic misreading.
        if out["cpu_proc_pct"]:
            out["cpu_proc_pct_of_board"] = {
                k: v / self._n_cores for k, v in out["cpu_proc_pct"].items()}
        return out

    def print_summary(self) -> None:
        s = self.summary()
        if not s.get("n_samples"):
            print("\n--- Hardware --- (no samples)")
            return
        print(f"\n--- Hardware ({s['n_samples']} samples @ {s['interval_s']}s, "
              f"{s['n_cores']} cores) ---")

        def line(label, d, unit, fmt="{:.1f}"):
            if not d:
                return
            print(f"{label:<28} mean {fmt.format(d['mean'])}{unit}  "
                  f"p95 {fmt.format(d['p95'])}{unit}  "
                  f"max {fmt.format(d['max'])}{unit}")

        line("CPU (process, 1 core=100%)", s["cpu_proc_pct"], "%")
        line("CPU (process, whole board)", s.get("cpu_proc_pct_of_board"), "%")
        line("CPU (system total)", s["cpu_sys_pct"], "%")
        line("RSS (process RAM)", s["rss_mb"], " MB")
        line("RAM used (system)", s["mem_used_pct"], "%")
        line("SoC temperature", s["temp_c"], " C")
        line("CPU frequency", s["freq_mhz"], " MHz")
        if s.get("throttle_flags_seen"):
            print(f"!! THROTTLING DETECTED: "
                  f"{', '.join(sorted(s['throttle_flags_seen']))}")


# --------------------------------------------------------------------------- #
# 4. BCI throughput
# --------------------------------------------------------------------------- #
def wolpaw_itr(accuracy: float, n_classes: int, decisions_per_min: float) -> dict:
    """
    Wolpaw Information Transfer Rate.

        B = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))   [bits / decision]
        ITR = B * decisions_per_min                          [bits / min]

    Reporting ITR alongside accuracy is what lets you argue that a slower but
    more accurate model is (or is not) the better real-time choice.
    """
    p = float(np.clip(accuracy, 0.0, 1.0))
    n = int(n_classes)
    if n < 2:
        return {"bits_per_decision": 0.0, "bits_per_min": 0.0}
    if p <= 1.0 / n:
        bits = 0.0
    elif p >= 1.0:
        bits = float(np.log2(n))
    else:
        bits = (np.log2(n) + p * np.log2(p)
                + (1 - p) * np.log2((1 - p) / (n - 1)))
        bits = float(max(bits, 0.0))
    return {"bits_per_decision": bits,
            "bits_per_min": bits * float(decisions_per_min)}


# --------------------------------------------------------------------------- #
# 5. Report I/O
# --------------------------------------------------------------------------- #
def write_report(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(payload, indent=2, default=default))
    return path


__all__ = [
    "StageTimer", "HardwareMonitor", "system_info", "wolpaw_itr",
    "read_temperature_c", "read_cpu_freq_mhz", "read_throttled",
    "write_report", "nullcontext",
]