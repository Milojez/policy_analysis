#This script generates the figure 9. please run it from /CatS folder
import matplotlib.pyplot as plt

from cats_processing import load_cats_csv, dialpos_to_bw_rank, cats_to_CatHb
from signal_for_dial import build_signals_all_videos
from fig9_metrics import compute_fig9_curves

# ------------------ USER SETTINGS ------------------
# CATS_CSV = "./Cats_new_python/CatS_reproduced_python.csv"  #path to the csv file
# SAVE_FIG = "./Figure_9/fig9_CatS_reproduced_python.png"         #"fig9.png" or None, ( "./Figure_9/fig9_CatS_reproduced_python.png" )
# PLOT_TITLE = "Figure 9 (CatS Reproduced): AOI vs Angle, Velocity, and Time to Crossing"

# CATS_CSV = "./Cats_org/CatS_org.csv"
# SAVE_FIG = "./Figure_9/fig9_CatS_org.png"  
# PLOT_TITLE = "Figure 9 (CatS Original): AOI vs Angle, Velocity, and Time to Crossing"

CATS_CSV = "./Cats_from_fixations/CatS_from_fixations_noDial7.csv"
SAVE_FIG = "./Figure_9/fig9_CatS_from_fixations.png"  
PLOT_TITLE = "Figure 9 (CatS from Fixations): AOI vs Angle, Velocity, and Time to Crossing"

CATS_ARE_BW = False     # True if cat already = bandwidth rank (1..6)
USE_MATLAB_STD = True   # True to match MATLAB std() behavior

# ---------------------------------------------------

# 1) load CatS
CatS = load_cats_csv(CATS_CSV)

# 2) convert to bandwidth rank if needed
if CATS_ARE_BW:
    CatSb = CatS
else:
    CatSb = dialpos_to_bw_rank(CatS)

# 3) percent participants per frame per bandwidth
CatHb = cats_to_CatHb(CatSb)

# 4) build signals (signal_for_dial port)
signals = build_signals_all_videos(t_end=90.0, use_matlab_std=USE_MATLAB_STD)

# 5) compute Figure 9 curves
angle_c, vel_c, ttc_c, NSI, NVSI, NTSI = compute_fig9_curves(signals, CatHb, fps=50)

# 6) plot
labels = ["0.03 Hz","0.05 Hz","0.12 Hz","0.20 Hz","0.32 Hz","0.48 Hz"]

plt.figure(figsize=(14,4))
plt.suptitle(PLOT_TITLE, fontsize=14)

ax1 = plt.subplot(1,3,1)
for bw in range(5, -1, -1):
    ax1.plot(angle_c, NSI[:, bw], linewidth=2, label=labels[bw])
ax1.set_xlabel("Pointer angle (deg)")
ax1.set_ylabel("Percent time on AOI (%)")
ax1.set_xlim(-120, 120)
ax1.set_ylim(0, 45)
ax1.grid(True)
ax1.legend()

ax2 = plt.subplot(1,3,2)
for bw in range(5, -1, -1):
    ax2.plot(vel_c, NVSI[:, bw], linewidth=2)
ax2.set_xlabel("Pointer velocity (deg/s)")
ax2.set_xlim(-100, 100)
ax2.set_ylim(0, 45)
ax2.grid(True)
ax2.set_yticklabels([])

ax3 = plt.subplot(1,3,3)
for bw in range(5, -1, -1):
    ax3.plot(ttc_c, NTSI[:, bw], linewidth=2, label=labels[bw])
ax3.set_xlabel("Time to crossing (s)")
ax3.set_xlim(-10, 10)
ax3.set_ylim(0, 45)
ax3.grid(True)
ax3.set_yticklabels([])
ax3.legend(loc="upper right")

plt.tight_layout()
if SAVE_FIG:
    plt.savefig(SAVE_FIG, dpi=200, bbox_inches="tight")
plt.show()

