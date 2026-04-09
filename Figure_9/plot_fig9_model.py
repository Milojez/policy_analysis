# Generates Figure 9 from the PolicyNet's predicted CatS for video 7.
# Mirrors plot_fig9.py but restricts computation to video 7 only,
# since predictions only cover that video.
#
# Run after:
#   1. generate_predictions.py
#   2. predictions_to_Cats.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # policy_analysis/
sys.path.insert(0, _HERE)                           # Figure_9/

import config
from cats_processing import load_cats_csv, dialpos_to_bw_rank
from signal_for_dial import signal_for_dial

TARGET_VIDEO = 7          # must match predictions_to_Cats.py
VIDEO_IDX    = TARGET_VIDEO - 1   # 0-based index into CatS arrays

CATS_CSV = os.path.join(
    config.RESULTS, "predictions",
    f"CatS_from_predicted_fixations_video{TARGET_VIDEO}.csv",
)
SAVE_FIG = os.path.join(
    config.RESULTS,
    f"fig9_model_predictions_video{TARGET_VIDEO}.png",
)
PLOT_TITLE = (
    f"Figure 9 (PolicyNet predictions, video {TARGET_VIDEO}) — {config.MODEL_NAME}: "
    "AOI vs Angle, Velocity, and Time to Crossing"
)

CATS_ARE_BW    = False
USE_MATLAB_STD = True

# ── 1. Load CatS and convert to bandwidth rank ─────────────────────────────────

CatS  = load_cats_csv(CATS_CSV)           # (92, 7, 4500)
CatSb = dialpos_to_bw_rank(CatS)          # same shape, dial pos → bw rank

# ── 2. CatHb for video 7 only ─────────────────────────────────────────────────
# CatHb_v7[frame, bw] = % participants looking at that bw at that frame

X        = CatSb[:, VIDEO_IDX, :]         # (92, 4500)
valid_pp = np.any(~np.isnan(X), axis=1)
n_valid  = valid_pp.sum()
print(f"Valid participants for video {TARGET_VIDEO}: {n_valid}")

vals     = X[valid_pp, :]                  # (n_valid, 4500)
CatHb_v7 = np.full((4500, 6), np.nan)
for bw in range(1, 7):
    CatHb_v7[:, bw - 1] = 100.0 * np.nanmean(vals == bw, axis=0)

# ── 3. Signal for video 7 only ─────────────────────────────────────────────────

sig_v7 = signal_for_dial(90.0, rg=TARGET_VIDEO, use_matlab_std=USE_MATLAB_STD)
# sig_v7: (4500, 6) radians, bw low→high

# ── 4. Compute Figure 9 curves (single-video version) ─────────────────────────

fps   = 50
temp2 = sig_v7                                          # (4500, 6) angle
temp3 = CatHb_v7                                        # (4500, 6) % on AOI
temp4 = np.vstack([np.full((1, 6), np.nan),
                   np.diff(temp2, axis=0) * fps])       # (4500, 6) velocity rad/s
temp5 = temp2 / (-temp4)                               # (4500, 6) TTC

IV  = np.arange(-np.pi, np.pi + np.deg2rad(5), np.deg2rad(5))
IV3 = np.arange(-100, 100.5, 0.5)

NSI  = np.full((len(IV)  - 1, 6), np.nan)
NVSI = np.full((len(IV)  - 1, 6), np.nan)
NTSI = np.full((len(IV3) - 1, 6), np.nan)

MIN_SAMPLES = 50   # lower threshold than original 250 — only 1 video here

for bw in range(6):
    for i in range(len(IV) - 1):
        lo, hi = IV[i], IV[i + 1]
        idx_a = np.where((temp2[:, bw] > lo) & (temp2[:, bw] <= hi))[0]
        idx_v = np.where((temp4[:, bw] > lo) & (temp4[:, bw] <= hi))[0]
        if idx_a.size > MIN_SAMPLES:
            NSI[i, bw]  = np.nanmean(temp3[idx_a, bw])
        if idx_v.size > MIN_SAMPLES:
            NVSI[i, bw] = np.nanmean(temp3[idx_v, bw])

    for i in range(len(IV3) - 1):
        lo, hi = IV3[i], IV3[i + 1]
        idx_t = np.where((temp5[:, bw] > lo) & (temp5[:, bw] <= hi))[0]
        if idx_t.size > MIN_SAMPLES:
            NTSI[i, bw] = np.nanmean(temp3[idx_t, bw])

angle_c = np.rad2deg(IV[:-1]  + np.diff(IV)  / 2)
vel_c   = angle_c.copy()
ttc_c   = IV3[:-1] + np.diff(IV3) / 2

# ── 5. Plot ────────────────────────────────────────────────────────────────────

labels = ["0.03 Hz", "0.05 Hz", "0.12 Hz", "0.20 Hz", "0.32 Hz", "0.48 Hz"]

plt.figure(figsize=(14, 4))
plt.suptitle(PLOT_TITLE, fontsize=13)

ax1 = plt.subplot(1, 3, 1)
for bw in range(5, -1, -1):
    ax1.plot(angle_c, NSI[:, bw], linewidth=2, label=labels[bw])
ax1.set_xlabel("Pointer angle (deg)")
ax1.set_ylabel("Percent time on AOI (%)")
ax1.set_xlim(-120, 120)
ax1.set_ylim(0, 45)
ax1.grid(True)
ax1.legend()

ax2 = plt.subplot(1, 3, 2)
for bw in range(5, -1, -1):
    ax2.plot(vel_c, NVSI[:, bw], linewidth=2)
ax2.set_xlabel("Pointer velocity (deg/s)")
ax2.set_xlim(-100, 100)
ax2.set_ylim(0, 45)
ax2.grid(True)
ax2.set_yticklabels([])

ax3 = plt.subplot(1, 3, 3)
for bw in range(5, -1, -1):
    ax3.plot(ttc_c, NTSI[:, bw], linewidth=2, label=labels[bw])
ax3.set_xlabel("Time to crossing (s)")
ax3.set_xlim(-10, 10)
ax3.set_ylim(0, 45)
ax3.grid(True)
ax3.set_yticklabels([])
ax3.legend(loc="upper right")

plt.tight_layout()
os.makedirs(config.RESULTS, exist_ok=True)
plt.savefig(SAVE_FIG, dpi=200, bbox_inches="tight")
print(f"Saved: {SAVE_FIG}")
plt.show()
