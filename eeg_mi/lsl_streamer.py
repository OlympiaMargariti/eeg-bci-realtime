import time
import argparse
from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
import joblib
from pylsl import StreamInfo, StreamOutlet

from . import config as C
from .preprocessing import load_raw

TARGET_TRIGGERS = (71, 121, 74, 124)     # classification classes - change accordingly


def _ci_find(directory: Path, name: str) -> Path:
    """Case-insensitive lookup ('Pilot1_*' vs 'pilot18_*' work on Windows only)."""
    exact = Path(directory) / name
    if exact.exists():
        return exact
    low = name.lower()
    for p in Path(directory).iterdir():
        if p.name.lower() == low:
            return p
    raise FileNotFoundError(f"{name} not found in {directory}")


@lru_cache(maxsize=1)
def _load_recording(bdf_file: str):
    """Loaded once per subject: a batch over several models re-uses it."""
    raw = load_raw(bdf_file)
    # Locating events before isolating eeg channel and thus throwing the status channel away
    events = mne.find_events(raw, stim_channel='Status', verbose="ERROR")
    sfreq = raw.info['sfreq']
    raw.pick_types(eeg=True)
    return raw.get_data(), list(raw.ch_names), events, sfreq


def build_ground_truth(events, n_samples, sfreq, test_times_sec=None,
                       bundle_sfreq=None, match_tolerance_sec=0.15,
                       tmin=C.TMIN, tmax=C.TMAX):
    """
    GroundTruth channel: -1 everywhere except inside the labelled window of
    every (test-set) trial: 0.0 -> movement, 1.0 -> rest.

    When the bundle's test timestamps are known, each trial's window is
    defined on the MODEL's 128 Hz grid (cue sample e128 from the bundle):
        samples  decim*(e128 + tmin*128)  ...  decim*(e128 + tmax*128)   (inclusive)
    i.e. the last labelled input sample is the one that becomes the LAST
    sample of the offline epoch after decimation, so the receiver's window
    (last WINDOW_LEN seconds at the instant the label ends) is the same
    window that replay_test.py verified against the offline epochs.

    Returns (ground_truth, found_trials, used_trials).
    """
    ground_truth = np.full(n_samples, -1.0)
    test_times_sec = None if test_times_sec is None else np.asarray(test_times_sec)
    found = used = 0

    for event in events:
        sample, trigger = int(event[0]), int(event[2])
        if trigger not in TARGET_TRIGGERS:
            continue
        found += 1
        event_time_sec = sample / sfreq

        if test_times_sec is not None:
            diffs = np.abs(test_times_sec - event_time_sec)
            if diffs.min() > match_tolerance_sec:
                continue  # not in the test set -> stays -1.0 (no data leakage)
            if bundle_sfreq:
                m = int(np.argmin(diffs))
                e128 = int(round(test_times_sec[m] * bundle_sfreq))
                decim = int(round(sfreq / bundle_sfreq))
                start_idx = decim * (e128 + int(round(tmin * bundle_sfreq)))
                end_idx = decim * (e128 + int(round(tmax * bundle_sfreq))) + 1
            else:
                start_idx = sample + int(tmin * sfreq)
                end_idx = sample + int(tmax * sfreq) + 1
        else:
            start_idx = sample + int(tmin * sfreq)
            end_idx = sample + int(tmax * sfreq) + 1

        used += 1
        class_val = 0.0 if trigger in (71, 121) else 1.0
        start_idx = max(0, min(start_idx, n_samples))
        end_idx = max(0, min(end_idx, n_samples))
        ground_truth[start_idx:end_idx] = class_val

    return ground_truth, found, used


def start_stream(bdf_file, bundle_path=None, match_tolerance_sec=0.15,
                 stream_name="BioSemi_Sim", wait_consumer=0.0, max_sec=None,
                 abort_on_consumer_loss=0.0, source_id=None):
    """
    Streaming the whole .bdf file utilizing LSL in order to simulate
    real-time data acquisition on a given sampling rate (512 Hz).
    The Ground Truth (status) channel is -1 for any trial not represented by
    the model classes, as well as trials that are part of the Train Set
    (avoiding data leakage), so the online accuracy is computed on the test
    set only. Trial <-> test-set synchronisation is done by timestamps.

    Batch / coordination options (all off by default = old behaviour):
      stream_name            : LSL name; the Pi worker discovers jobs by it
      wait_consumer          : seconds to wait for a receiver to connect BEFORE
                               any sample is sent (so the receiver always sees
                               sample 0 -- decimation phase and window grid
                               are then deterministic)
      max_sec                : stop after this many seconds of streaming
                               (profiling does not need the whole recording)
      abort_on_consumer_loss : seconds without any consumer after which the
                               stream is aborted (Pi crashed / rebooted)

    Returns one of: "done", "no_consumer", "consumer_lost", "interrupted".
    """
    print(f"Loading the EEG file: {bdf_file}...")
    eeg, eeg_names, events, sfreq = _load_recording(str(bdf_file))
    n_samples = eeg.shape[1]

    # Loading Test Set Trials' timestamps from the bundle
    test_times_sec, bundle_sfreq = None, None
    if bundle_path is not None:
        bundle = joblib.load(bundle_path)
        test_times_sec = bundle.get("test_event_times_sec", None)
        bundle_sfreq = bundle.get("sfreq", None)
        if not test_times_sec:
            print("[WARNING] 'test_event_times_sec' were not found "
                  "in the bundle -- ALL of the trials will be evaluated.")
            test_times_sec = None
        else:
            print(f"Loaded {len(test_times_sec)} test-set trial timestamps "
                  f"from the bundle.")

    ground_truth, found_trials, used_trials = build_ground_truth(
        events, n_samples, sfreq, test_times_sec, bundle_sfreq, match_tolerance_sec)

    if test_times_sec is not None:
        print(f"Found {found_trials} target trials in total, "
              f"evaluating {used_trials} (test set only, "
              f"tolerance={match_tolerance_sec}s).")
    else:
        print(f"Found {found_trials} target trials. "
              f"evaluating all of them (invalid bundle_path).")

    # Concatenate Ground Truth as the last channel to be streamed
    data = np.vstack([eeg, ground_truth])
    ch_names = eeg_names + ["GroundTruth"]
    n_channels = len(ch_names)

    # Creating the LSL stream
    info = StreamInfo(name=stream_name, type='EEG', channel_count=n_channels,
                      nominal_srate=sfreq, channel_format='float32',
                      source_id=source_id or f"sim_{stream_name}")

    chns = info.desc().append_child("channels")
    for ch in ch_names:
        chns.append_child("channel").append_child_value("label", ch)

    outlet = StreamOutlet(info)
    print(f"The '{stream_name}' LSL stream has successfully begun at {sfreq} Hz.")

    if wait_consumer and wait_consumer > 0:
        print(f"Waiting up to {wait_consumer:.0f}s for a receiver to connect...")
        if not outlet.wait_for_consumers(wait_consumer):
            print("No receiver connected -- giving up on this stream.")
            return "no_consumer"
        time.sleep(1.0)           # let the inlet start pulling before sample 0
        print("Receiver connected, streaming.")
    print("Send Ctrl+C to terminate.")

    chunk_size = 16
    sent_samples = 0
    start_time = time.time()
    last_consumer_ok = time.time()
    last_check = 0.0
    status = "done"

    try:
        while sent_samples < data.shape[1]:
            now = time.time()
            elapsed_time = now - start_time
            if max_sec is not None and elapsed_time >= max_sec:
                print(f"Reached --max-sec {max_sec:.0f}s.")
                break

            if abort_on_consumer_loss and now - last_check >= 1.0:
                last_check = now
                if outlet.have_consumers():
                    last_consumer_ok = now
                elif now - last_consumer_ok > abort_on_consumer_loss:
                    print("Receiver disappeared -- aborting this stream.")
                    return "consumer_lost"

            target_samples = int(elapsed_time * sfreq)
            if target_samples > sent_samples:
                n_send = min(chunk_size, target_samples - sent_samples)
                chunk_data = data[:, sent_samples: sent_samples + n_send].T
                outlet.push_chunk(chunk_data.tolist())
                sent_samples += n_send

            time.sleep(0.005)

        time.sleep(2.0)           # let the receiver drain the last samples
        print("The stream has completed!")
    except KeyboardInterrupt:
        print("\nStreamer Terminated.")
        status = "interrupted"
    return status


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, default=1,
                        help="Subject index from 1 to 33 (e.g., 1 for Pilot1)")
    parser.add_argument("-m", "--model", type=int, choices=range(1, 7), default=3,
                        help="Model ID: 1=svm, 2=rf, 3=fbcsp, 4=mdm, 5=tangent_space, 6=fb_tangent")
    parser.add_argument("--name", type=str, default="BioSemi_Sim",
                        help="LSL stream name")
    parser.add_argument("--wait-consumer", type=float, default=0.0,
                        help="wait up to N s for a receiver before sending")
    parser.add_argument("--max-sec", type=float, default=None,
                        help="stop streaming after N seconds")
    args = parser.parse_args()

    model_name = C.MODEL_MAP[args.model]
    test_file = _ci_find(C.DATA_DIR, f"Pilot{args.subject}.bdf")
    model_file = _ci_find(C.OUTPUT_DIR, f"Pilot{args.subject}_{model_name}.joblib")

    print("Result:", start_stream(test_file, bundle_path=model_file,
                                  stream_name=args.name,
                                  wait_consumer=args.wait_consumer,
                                  max_sec=args.max_sec))