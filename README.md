# EEG Motor-Imagery Pipeline (Movement vs Rest) — Offline Training & Online Pi Inference via LSL

This project is still a work in progress.

A complete pipeline for a real-time motor-imagery BCI. It pre-processes BioSemi `.bdf` 
recordings with MNE, trains and compares six different classifiers per subject using 
sliding-window augmentation, and streams data via LSL for real-time online classification.

## Why these design choices

**Task = Motor Imagery, MOVEMENT vs REST (imaginary-motion visual & auditory conditions,**
**triggers 71/121 vs 74/124).** Motor imagery is the canonical paradigm behind neuro-rehabilitation 
and assistive neuroprosthetics. Classifying *intent to move* from *no intent (rest)* is the 
foundational control signal for an embedded BCI. To target a different condition, edit `EVENT_ID` 
in `config.py`.

**Channels — a small sensorimotor set: FC3, FC4, C3, Cz, C4, CP3, CP4.**
Imagery is dominated by the mu (8–13 Hz) and beta (13–30 Hz) event-related desynchronisation 
over the hand area. C3/C4 sit directly over it, and Cz + the FC/CP ring add spatial context. 
The channels are resolved directly by their 10-20 standard names. Using just 7 channels 
(instead of 64) drastically reduces computational cost on the Raspberry Pi.

**Split & Augmentation — Subject-Specific, Stratified Group Hold-Out.** 
EEG is highly subject-specific, so the realistic deployment scenario is a model calibrated per user. 
- **Strict Separation:** A stratified 75/25 train/test split is performed *at the trial level* 
before any windowing, ensuring zero data leakage (Test set trials never bleed into the Train set). 
- **Sliding Window:** Trials are augmented into 1-second overlapping windows (0.25s step) to match the 
continuous stream format the classifier will see online, drastically increasing the training data volume.

**Models (6 competing pipelines).** 
1. `svm` & `rf`: Standard CSP + log-variance band-power features.
2. `fbcsp`: Filter Bank CSP (multiple frequency bands) + Mutual Info feature selection + SVM.
3. `mdm`: Riemannian MDM operating directly on trial covariance matrices.
4. `tangent_space`: Covariances mapped to the Tangent Space + Logistic Regression.
5. `fb_tangent`: Filter Bank Tangent Space + Feature Selection + PCA + Ridge (Highest performance potential).

## Preprocessing steps (`preprocessing.py`)
All preprocessing is designed to be causal and hardware-friendly (IIR/SOS filters instead of FIR) so 
the online inference perfectly matches the offline training.
1. Load `.bdf`, tag EEG and EOG channels.
2. Detrend / DC removal (mean subtraction).
3. Notch 50 Hz (+ 100 Hz harmonic).
4. **Causal Broadband Band-Pass 1–40 Hz** (replaces the narrow 8-30Hz filter to allow Filter Banks to
 extract low/high frequency features).
5. Common-Average Re-reference (CAR).
6. Downsample 512 → 128 Hz.
7. Epoching (0.5–3.0 s) without baseline correction.

## Offline Training
```bash
pip install -r requirements.txt
# edit DATA_DIR in eeg_mi/config.py to point at your Raw folder
python -m eeg_mi.train            # Run on all subjects
python -m eeg_mi.train --limit 3  # Quick test on the first 3 files
```
Outputs go to `eeg_mi/artifacts/`. The script saves one `PilotN_<model>.joblib` bundle **for each** 
trained model, plus a `summary.json` with per-subject and across-subject aggregated metrics.

## Online Inference via LSL
The real-time evaluation uses Lab Streaming Layer (LSL) to simulate the physical BCI board. The streamer 
plays back the `.bdf` file while the receiver buffers the data, applies causal preprocessing, and runs 
the saved pipeline.

To avoid artificial inflation of online accuracy, the streamer utilizes the test-set timestamps saved in 
the `.joblib` bundle. It dynamically flags the `GroundTruth` channel so that the receiver only evaluates 
accuracy on unseen (Test Set) trials.

**Terminal 1 (Streamer):**
```bash
# Starts the LSL stream for a given subject and marks Test-Set trials. When running configure subject and
# model, e.g. for Pilot1 and model fbcsp:
python -m eeg_mi.streamer -s 1 -m 3
```

**Terminal 2 OR Raspberry Pi (Receiver):**
```bash
# Connects to the LSL stream, applies causal filters, and predicts on-the-fly. hen running configure subject and
# model, e.g. for Pilot1 and model fbcsp:
python -m eeg_mi.receiver -s 1 -m 3
```
The receiver prints real-time status (Latency, CPU/RAM usage, Prediction vs Ground Truth) and computes 
the final online accuracy metric upon termination.
