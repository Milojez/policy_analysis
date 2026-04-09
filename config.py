"""
config.py
=========
Configuration for the PolicyNet: next-dial prediction from fixation history + signal history.
"""

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Raw data ───────────────────────────────────────────────────────────────────
GT_CSV  = os.path.join(_ROOT, "data", "GT_all_fixation_noDial7_noPPs.csv")
SIG_DIR = os.path.join(_ROOT, "data", "movie_signal_csv")

# Train: videos 1-5, Val: video 6, Test: video 7
TRAIN_VIDEOS = [1, 2, 3, 4, 5]
VAL_VIDEOS   = [6]
TEST_VIDEOS  = [7]

# ── AOI bounding boxes (x_min, x_max, y_min, y_max) ─────────────────────────
# From heatmap_ar_model/test_participant_analysis.py
# Boundary: x_min < x_fix <= x_max  and  y_min <= y_fix < y_max
AOI_BY_POSITION = {
    1: (116,  536,    1,  421),
    2: (750,  1170,   1,  421),
    3: (1385, 1805,   1,  421),
    4: (116,  536,  659, 1079),
    5: (750,  1170, 659, 1079),
    6: (1385, 1805, 659, 1079),
}

# ── Signal sampling ───────────────────────────────────────────────────────────
SIGNAL_LENGTH_S = 2     # seconds of signal history before next fixation
SIGNAL_HZ       = 10   # evenly-spaced frames per second to extract
SIGNAL_FRAMES   = int(round(SIGNAL_LENGTH_S * SIGNAL_HZ))  # = 20 frames

# Future signal window (during predicted fixation) for temporal conditioning
FUTURE_SIGNAL_S = 0.5
FUTURE_FRAMES   = int(round(FUTURE_SIGNAL_S * SIGNAL_HZ))  # = 5 frames

# ── Model ─────────────────────────────────────────────────────────────────────
PAST_FIXATIONS = 2       # past fixation AOI labels fed to GRU
STRIDE         = 1       # samples advance by this many fixations per step
HIDDEN_DIM     = 64
EMB_DIM        = 16      # AOI embedding dimension
SIG_FEAT       = 5       # signal features per (dial, frame): sin, cos, urgency, distance, speed_norm

# Ablation flags — set False to zero-out that branch (no rebuild needed)
USE_FIXATIONS = True
USE_SIGNAL    = True

# ── Regularisation ────────────────────────────────────────────────────────────
DROPOUT = 0.2
FIX_BRANCH_DROPOUT = 0.5   # prob of zeroing fixation branch (forces signal reliance)

# ── Loss ──────────────────────────────────────────────────────────────────────
LAMBDA_TEMPORAL = 0.05

# ── Training ──────────────────────────────────────────────────────────────────
SEED          = 42
EPOCHS        = 40
BATCH_SIZE    = 256
LR            = 1e-3
LR_MIN        = 1e-5
WARMUP_EPOCHS = 3
WEIGHT_DECAY  = 1e-4

# ── Transformer ───────────────────────────────────────────────────────────────
NHEAD           = 4
FF_DIM          = HIDDEN_DIM
N_SIGNAL_LAYERS = 1

# ── Dataset paths — named after parameters that affect sample construction ────
# Changing SIGNAL_LENGTH_S, PAST_FIXATIONS, STRIDE, SIGNAL_HZ, or FUTURE_SIGNAL_S
# requires re-running build_dataset.py.
DATASET_NAME       = (f"sl{SIGNAL_LENGTH_S}s"
                      f"_pf{PAST_FIXATIONS}"
                      f"_str{STRIDE}"
                      f"_hz{SIGNAL_HZ}"
                      f"_fut{FUTURE_SIGNAL_S}s")
DATASETS_BUILT_DIR = os.path.join(_ROOT, "datasets_built", DATASET_NAME)

DATA_PKL = os.path.join(DATASETS_BUILT_DIR, "policy_dataset.pkl")
TRAIN_PT = os.path.join(DATASETS_BUILT_DIR, "policy_train.pt")
VAL_PT   = os.path.join(DATASETS_BUILT_DIR, "policy_val.pt")
TEST_PT  = os.path.join(DATASETS_BUILT_DIR, "policy_test.pt")
META_PKL = os.path.join(DATASETS_BUILT_DIR, "policy_meta.pkl")

# ── Outputs ───────────────────────────────────────────────────────────────────
MODEL_NAME = "policy_sepdur_h64_n2_str1_fbd05_hz10"
CKPT_DIR   = os.path.join(_ROOT, "checkpoints", MODEL_NAME)
CKPT       = os.path.join(CKPT_DIR, "policy_best.pth")
LOG_PATH   = os.path.join(CKPT_DIR, f"{MODEL_NAME}_log.csv")
RESULTS    = os.path.join(_ROOT, "results", MODEL_NAME)
