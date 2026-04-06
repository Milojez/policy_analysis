"""
config.py
=========
Configuration for the PolicyNet: next-dial prediction from fixation history + signal history.
"""

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Data ──────────────────────────────────────────────────────────────────────
GT_CSV  = os.path.join(_ROOT, "data", "GT_all_fixation_noDial7_noPPs.csv")
SIG_DIR = os.path.join(_ROOT, "data", "movie_signal_csv")
DATA_PKL  = os.path.join(_ROOT, "data", "policy_dataset.pkl")   # full pkl (for visualize)
TRAIN_PT  = os.path.join(_ROOT, "data", "policy_train.pt")
VAL_PT    = os.path.join(_ROOT, "data", "policy_val.pt")
TEST_PT   = os.path.join(_ROOT, "data", "policy_test.pt")
META_PKL  = os.path.join(_ROOT, "data", "policy_meta.pkl")     # stats only (small)

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
SIGNAL_LENGTH_S = 2   # seconds of signal history before next fixation
SIGNAL_HZ       = 10    # evenly-spaced frames per second to extract
# F = round(SIGNAL_LENGTH_S * SIGNAL_HZ) = 20 total frames per sample
# number of frames per signal window after sampling
SIGNAL_FRAMES = int(round(SIGNAL_LENGTH_S * SIGNAL_HZ))
# ── Model ─────────────────────────────────────────────────────────────────────
PAST_FIXATIONS = 2       # past fixation AOI labels fed to GRU
STRIDE         = 1       # samples per (pp, video) advance by this many fixations
                         # STRIDE=1 → consecutive samples share PAST_FIXATIONS-1 context fixations
                         # STRIDE=PAST_FIXATIONS → no overlap (fully non-overlapping windows)
HIDDEN_DIM     = 64
EMB_DIM        = 16      # AOI embedding dimension
SIG_FEAT       = 5       # signal features per (dial, frame): sin, cos, urgency, distance, speed_norm

# Ablation flags — set False to zero-out that branch (no rebuild needed)
USE_FIXATIONS = True     # use past fixation AOI history
USE_SIGNAL    = True     # use recent dial signal history

# ── Regularisation ────────────────────────────────────────────────────────────
DROPOUT = 0.2            # dropout probability in fusion MLP
# Probability of zeroing out past_aois for a training sample (forces signal use)
FIX_BRANCH_DROPOUT = 0.5

# ── Loss ──────────────────────────────────────────────────────────────────────
# Weight of temporal (saccade + duration) MSE loss vs dial CrossEntropy
LAMBDA_TEMPORAL = 0.05

# ── Training ──────────────────────────────────────────────────────────────────
SEED         = 42
EPOCHS       = 40
BATCH_SIZE   = 256
LR           = 1e-3
LR_MIN       = 1e-5   # cosine annealing floor
WARMUP_EPOCHS = 3     # linear warmup before cosine decay begins
WEIGHT_DECAY = 1e-4

# ── Transformer ───────────────────────────────────────────────────────────
NHEAD = 4
FF_DIM = HIDDEN_DIM        # lean FFN — keeps transformer ~99k params at this data scale
N_SIGNAL_LAYERS = 1        # one layer sufficient for cross-token interaction

# ── Outputs ───────────────────────────────────────────────────────────────────
MODEL_NAME = "policy_h64_n2_str1_fbd05_hz10"
CKPT_DIR   = os.path.join(_ROOT, "checkpoints", MODEL_NAME)
CKPT       = os.path.join(CKPT_DIR, "policy_best.pth")
LOG_PATH   = os.path.join(CKPT_DIR, f"{MODEL_NAME}_log.csv")
RESULTS    = os.path.join(_ROOT, "results", MODEL_NAME)
