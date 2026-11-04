"""
Central configuration for the EEG Motor-Imagery pipeline:

    - Configure the path where the eeg .bdf files are located.
    - Target task (chosen for max biomedical relevance for a real-time BCI):
        Motor Imagery -- MOVEMENT vs REST
        using the "imaginary motion (visual + auditory)" condition of the dataset.
    - Epoching parameters -- trail duration as well as window lenght and step
    - Filter bands
    - Channels used for prediction
    - Model mapping to simplify model selection in the online interface

"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. Data location
# --------------------------------------------------------------------------- #
# Folder that holds subject .bdf files
DATA_DIR = Path(r"C:\Users\user\Downloads\Raw\Raw")
FILE_GLOB = "Pilot*.bdf"          # how the recordings are named

# Where trained models / reports are written.
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"

# --------------------------------------------------------------------------- #
# 2. Recording layout (from the experimental-design notes)
# --------------------------------------------------------------------------- #
SFREQ_ORIG = 512                  # original sampling rate (Hz)
N_EEG = 64                        # channels 1-64 are EEG
# channels 65-72 are EOG / leg-EMG -> used for artifact handling, not features
EOG_CHANNELS_IDX = [64, 65, 66, 67]   # 0-based: 65-68 = VEOG/HEOG
STIM_CHANNEL = "Status"           # BioSemi trigger channel

# --------------------------------------------------------------------------- #
# 3. The classification problem
# --------------------------------------------------------------------------- #
# Triggers for "imaginary motion (visual)": 71 both / 72 right / 73 left / 74 none
# Triggers for "imaginary motion (auditory)": 121 both / 122 right / 123 left / 124 none
# We classify the three most informative, control-relevant classes.

EVENT_ID = {
    "movement":  [71, 121],   # mi both
    "rest":  [74, 124],   # mi rest
}

# Trial timing (seconds) relative to the cue trigger.
# Each trial lasts 3 s; we skip the first 0.5 s (cue/visual onset transient).
TMIN = 0.5
TMAX = 2
BASELINE = None                   # no baseline correction (online-friendly)

# Parameters for Sliding Window Cropping, resulting in more epochs (training data)
WINDOW_LEN = 1.5
WINDOW_STEP = 0.25

# --------------------------------------------------------------------------- #
# 4. Preprocessing parameters
# --------------------------------------------------------------------------- #
SFREQ_TARGET = 128                # downsample target (Hz). 128 keeps <=64 Hz info
NOTCH_FREQ = 50                   # mains frequency in Greece/EU (Hz)
BANDPASS = (3.0, 30.0)            # mu (8-13) + beta (13-30): the MI-relevant band
BROAD_BANDPASS = (1.0, 40.0)      # broad band used *before* ICA
USE_ICA = False                   # remove EOG (blink/saccade) components
ICA_N_COMPONENTS = 0.95          # ICA components to keep (PCA variance ratio)
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# 5. Channel selection (small, motor-cortex set)
# --------------------------------------------------------------------------- #
# Sensorimotor channels over the hand area. C3/C4 capture contralateral
# mu/beta ERD for right/left imagery; Cz + the FC/CP ring add spatial context.
MOTOR_CHANNELS_1020 = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
# MOTOR_CHANNELS_1020 = [
#     "FC3", "FC1", "FCz", "FC2", "FC4", 
#     "C3", "C1", "Cz", "C2", "C4", 
#     "CP3", "CP1", "CPz", "CP2", "CP4"
# ]

# --------------------------------------------------------------------------- #
# 6. Model / evaluation
# --------------------------------------------------------------------------- #
N_CSP_COMPONENTS = 6             # spatial filters for SVM/RF feature path
TEST_SIZE = 0.25                 # within-subject hold-out fraction
CV_FOLDS = 5                     # stratified cross validation on the training portion

# --------------------------------------------------------------------------- #
# 7. CLI / Pipeline Mapping
# --------------------------------------------------------------------------- #
MODEL_MAP = {
    1: "svm",
    2: "rf",
    3: "fbcsp",
    4: "mdm",
    5: "tangent_space",
    6: "fb_tangent"
}