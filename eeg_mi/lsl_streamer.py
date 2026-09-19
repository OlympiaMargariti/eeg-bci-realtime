import time
import mne
import numpy as np
import joblib
import argparse
from pylsl import StreamInfo, StreamOutlet
from . import config as C
from .preprocessing import load_raw


def start_stream(bdf_file, bundle_path=None, match_tolerance_sec=0.15):
    """
    Streaming the whole .bdf file utilizing LSL in order to simulate 
    real-time data acquisition on a given sampling rate (512 Hz).
    The Ground Truth (status) channel is modified to be -1 for any trial 
    not represented by the model classes, as well as trials  that are part
    of the Train Set avoiding Data Leakage. This way the online accuracy
    is computated based only on the Test Set predictions. The trial <-> test-set
    synchrozination is defined by timestamps (seconds) as frequency would not
    be a reliable factor (downsampling before saving the bundle).

    """
    print(f"Loading the EEG file: {bdf_file}...")
    raw = load_raw(bdf_file)

    # Locating events before isolating eeg channel and thus throughing the status channel away
    events = mne.find_events(raw, stim_channel='Status', verbose="ERROR")
    sfreq = raw.info['sfreq']

    # Isolating only eeg channels to be streamed
    raw.pick_types(eeg=True)
    data = raw.get_data()
    n_samples = data.shape[1]

    # Classification classes - to be changed accoringly
    target_triggers = [71, 121, 74, 124]

    # Loading Test Set Trias' timestamps from the bundle
    test_times_sec = None
    if bundle_path is not None:
        bundle = joblib.load(bundle_path)
        test_times_sec = bundle.get("test_event_times_sec", None)
        if not test_times_sec:
            print("[WARNING] 'test_event_times_sec' were not found"
                  "in the bundel -- ALL of the trials will be evaluated.")
            test_times_sec = None
        else:
            test_times_sec = np.array(test_times_sec)
            print(f"Loaded {len(test_times_sec)} test-set trial timestamps "
                  f"from the bundle.")

    # Initializing the whole Ground Truth (status) channel to -1 (Invalid Trial)
    ground_truth = np.full(n_samples, -1.0)

    found_trials = 0
    used_trials = 0
    for event in events:
        sample = event[0]
        trigger = event[2]

        if trigger in target_triggers:
            found_trials += 1
            event_time_sec = sample / sfreq

            # Checking if this trial is a part of the Test-Set Trials
            # based on the timestamp (given a resampling rounding tolerance)
            if test_times_sec is not None:
                diffs = np.abs(test_times_sec - event_time_sec)
                if diffs.min() > match_tolerance_sec:
                    continue  # not on Test Set -> Ground Truth remains -1.0

            used_trials += 1
            # 0.0 -> movement, 1.0 -> rest -- Change accordingly
            class_val = 0.0 if trigger in [71, 121] else 1.0

            # Initializing true start and end time of the event
            # according to definitions in config.py
            start_idx = sample + int(C.TMIN * sfreq)
            end_idx = sample + int(C.TMAX * sfreq)

            # Ensuring compatibility with the file's limits
            start_idx = max(0, min(start_idx, n_samples))
            end_idx = max(0, min(end_idx, n_samples))

            # Updating Ground truth channel
            ground_truth[start_idx:end_idx] = class_val

    if test_times_sec is not None:
        print(f"Found {found_trials} target trials in total, "
              f"evaluating {used_trials} (test set only, "
              f"tolerance={match_tolerance_sec}s).")
    else:
        print(f"Found {found_trials} target trials. "
              f"evaluating all of them (invalid bundle_path).")

    # Concatenate Ground Truth as the last channel to be streamed
    data = np.vstack([data, ground_truth])
    ch_names = raw.ch_names + ["GroundTruth"]
    n_channels = len(ch_names)

    # Creating the LSL stream
    info = StreamInfo(name='BioSemi_Sim', type='EEG', channel_count=n_channels,
                       nominal_srate=sfreq, channel_format='float32', source_id='sim_123')

    chns = info.desc().append_child("channels")
    for ch in ch_names:
        chns.append_child("channel").append_child_value("label", ch)

    outlet = StreamOutlet(info)
    print(f"The 'BioSemi_Sim' LSL stream has successfuly began at {sfreq} Hz.")
    print("Send Ctrl+C to terminate.")

    chunk_size = 16
    sent_samples = 0
    start_time = time.time()

    try:
        while sent_samples < data.shape[1]:
            elapsed_time = time.time() - start_time
            target_samples = int(elapsed_time * sfreq)

            if target_samples > sent_samples:
                n_send = min(chunk_size, target_samples - sent_samples)
                chunk_data = data[:, sent_samples: sent_samples + n_send].T
                outlet.push_chunk(chunk_data.tolist())
                sent_samples += n_send

            time.sleep(0.005)

        print("The stream has completed!")
    except KeyboardInterrupt:
        print("\nStreamer Terminated.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, default="1", 
                        help="Subject index from 1 to 33 (e.g., 1 for Pilot1)")
    parser.add_argument("-m", "--model", type=int, choices=range(1, 7), default=3, 
                        help="Model ID: 1=svm, 2=rf, 3=fbcsp, 4=mdm, 5=tangent_space, 6=fb_tangent")
    
    args = parser.parse_args()
    model_name = C.MODEL_MAP[args.model]
    
    test_file = C.DATA_DIR / f"Pilot{args.subject}.bdf"
    model_file = C.OUTPUT_DIR / f"Pilot{args.subject}_{model_name}.joblib"

    # Default file paths
    # test_file = C.DATA_DIR / "Pilot1.bdf"
    # model_file = C.OUTPUT_DIR / "Pilot1_fbcsp.joblib"

    start_stream(test_file, bundle_path=model_file)