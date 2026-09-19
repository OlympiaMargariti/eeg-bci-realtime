"""
Evaluating the model performance by making Real-Time predictions
on an LSL stream. The sliding window parameters are defined by 
config.py. Real-time hardware parameters are also computed during 
the simulation.
"""

import time
import os
import psutil
import numpy as np
import argparse
from pylsl import resolve_byprop, StreamInlet
from .online_inference import RealTimeMI
from . import config as C

def run_receiver(model_path):
    print("Searching fot an LSL EEG stream...")
    # In case more than one streams are currently transmitted, a specific name must be defined
    streams = resolve_byprop('type', 'EEG')
    if not streams:
        print("No 'EEG' LSL stream was found.")
        return
        
    inlet = StreamInlet(streams[0])
    
    info = inlet.info()
    sfreq_in = info.nominal_srate()
    print(f"Connected! Input sampling rate: {sfreq_in} Hz")
    
    # Locating the eeg channels by name
    ch_names = []
    ch = info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        ch_names.append(ch.child_value("label"))
        ch = ch.next_sibling()
        
    if not ch_names:
        print("WARNING: No EEG channel names were found on the LSL.")
        eeg_ch_names = None
    else:
        # Removing the last channel ("GroundTruth") so that the BCI object gets only EEG channels
        eeg_ch_names = ch_names[:-1]

    # Loading the main Pipeline
    bci = RealTimeMI(model_path, all_ch_names=eeg_ch_names, sfreq_in=sfreq_in)
    process = psutil.Process(os.getpid())
    
    print(f"Starting Inference (Window: {bci.win_sec}s, Step: {C.WINDOW_STEP}s)")
    
    last_predict_time = time.time()
    step_sec = C.WINDOW_STEP 
    
    correct_preds = 0
    total_preds = 0
    
    # List to keep track of the prediction latencies in seconds
    latencies_in_trial = []
    
    try:
        while True:
            chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=64)
            if chunk:
                samples = np.array(chunk).T # (n_channels, n_samples)
                
                # Splitting EEG and Ground Truth channels
                eeg_samples = samples[:-1, :]
                bci.push(eeg_samples)
                
                current_time = time.time()
                if (current_time - last_predict_time) >= step_sec:
                    
                    t0 = time.perf_counter()
                    pred = bci.try_predict()
                    t1 = time.perf_counter()
                    
                    if pred is not None:
                        latency_ms = (t1 - t0) * 1000.0
                        cpu_percent = process.cpu_percent()
                        mem_mb = process.memory_info().rss / (1024 * 1024)
                        
                        # Getting Ground Truth label of the last sample in current chunk
                        gt_val = samples[-1, -1] 

                        # When a valid Trial is found the window will be forwarded for predition
                        if gt_val != -1.0: 
                            gt_str = "movement" if gt_val == 0.0 else "rest"
                            total_preds += 1
                            
                            # Saving latency in ms
                            latencies_in_trial.append(latency_ms)
                            
                            is_correct = (pred == gt_str)
                            if is_correct:
                                correct_preds += 1
                                
                            acc_percent = (correct_preds / total_preds) * 100
                            status = "TRUE" if is_correct else "FALSE"
                            
                            print(f"Prediction: {pred:<8} | Real State: {gt_str:<8} [{status:<5}] | "
                                  f"Acc: {acc_percent:>5.1f}% | "
                                  f"Lat: {latency_ms:>4.1f} ms | CPU: {cpu_percent:>4.1f}% | RAM: {mem_mb:>5.1f} MB")
                        else:
                            # Non valid trial/padding (ignored by accuracy)
                            # To be commented out -- should this be taken into account for metric calculations??
                            print(f"Prediction: {pred:<8} | (Non Valid Trial)           | "
                                  f"Lat: {latency_ms:>4.1f} ms | CPU: {cpu_percent:>4.1f}% | RAM: {mem_mb:>5.1f} MB")
                            
                        last_predict_time = current_time
                        
    except KeyboardInterrupt:
        print("\nReceiver Terminated.")
        if total_preds > 0:
            # Calculating total max and avg latency in ms
            max_lat = max(latencies_in_trial)
            avg_lat = sum(latencies_in_trial) / len(latencies_in_trial)
            
            print(f"\n============= Final Results =============")
            print(f"Total prediction (Tets Set Trials): {total_preds}")
            print(f"Correct Predictions: {correct_preds}")
            print(f"Final Real-Time Accuracy: {(correct_preds / total_preds) * 100:.2f}%")
            print(f"Max prediction latency: {max_lat:.2f} ms")
            print(f"Mean prediction latency: {avg_lat:.2f} ms")
            print(f"===============================================\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, default="1", 
                        help="Subject index from 1 to 33 (e.g., 1 for Pilot1)")
    parser.add_argument("-m", "--model", type=int, choices=range(1, 7), default=3, 
                        help="Model ID: 1=svm, 2=rf, 3=fbcsp, 4=mdm, 5=tangent_space, 6=fb_tangent")
    
    args = parser.parse_args()
    model_name = C.MODEL_MAP[args.model]
    
    model_file = C.OUTPUT_DIR / f"Pilot{args.subject}_{model_name}.joblib"

    # Default model parameters' path
    #model_file = C.OUTPUT_DIR / "Pilot1_fbcsp.joblib"

    run_receiver(model_file)