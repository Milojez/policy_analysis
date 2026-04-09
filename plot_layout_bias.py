"""
plot_layout_bias.py
===================
Visualises the bandwidth-rank-per-position assignment across all 7 videos,
exposing the positional prior the model learns from training data and the
distribution shift in the test video.

Produces:
  results/{MODEL_NAME}/layout_bias.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Bandwidth rank per position per video ──────────────────────────────────────
# DIAL_CONFIG[v, p] encodes the layout for video v+1.
# dialpos_to_bw_rank logic: argsort gives position of each rank → invert to get
# rank of each position.

DIAL_CONFIG = np.array([
    [3, 6, 2, 4, 5, 1],   # video 1  (train)
    [4, 6, 1, 5, 2, 3],   # video 2  (train)
    [1, 3, 4, 5, 6, 2],   # video 3  (train)
    [5, 3, 2, 6, 1, 4],   # video 4  (train)
    [6, 2, 1, 3, 4, 5],   # video 5  (train)
    [3, 5, 2, 4, 1, 6],   # video 6  (val)
    [5, 1, 4, 3, 2, 6],   # video 7  (test)
], dtype=int)

n_videos, n_pos = DIAL_CONFIG.shape   # 7, 6

# rank_matrix[v, p] = bandwidth rank (1=lowest, 6=highest) of position p+1 in video v+1
rank_matrix = np.zeros((n_videos, n_pos), dtype=int)
for v in range(n_videos):
    b   = np.argsort(DIAL_CONFIG[v]) + 1          # b[bw-1] = dial_pos (1-indexed)
    inv = np.zeros(n_pos + 1, dtype=int)
    for bw in range(1, n_pos + 1):
        inv[b[bw - 1]] = bw                        # inv[dial_pos] = bw_rank
    for p in range(n_pos):
        rank_matrix[v, p] = inv[p + 1]

# ── Training / val / test split ───────────────────────────────────────────────
train_idx = [0, 1, 2, 3, 4]   # videos 1-5
val_idx   = [5]                # video 6
test_idx  = [6]                # video 7

train_avg = rank_matrix[train_idx].mean(axis=0)    # (6,)
train_std = rank_matrix[train_idx].std(axis=0)     # (6,) — low std = high positional bias
test_rank = rank_matrix[test_idx[0]]               # (6,)
shift     = test_rank - train_avg                  # signed shift per position

# Observed count ratios from diagnose_temporal.py (pred / GT)
# Update these if you rerun diagnose_temporal.py with a new model
count_ratio = np.array([0.88, 0.96, 1.24, 0.79, 1.04, 1.12])  # D1..D6

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'─'*95}")
print(f"  Bandwidth rank per dial position — bias & count ratio analysis")
print(f"{'─'*95}")
header = ("  Pos  " + "  ".join(f"V{v+1}" for v in range(n_videos)) +
          "  TrainAvg  TrainStd  TestRank  Shift  CountRatio  BiasLevel")
print(header)
print(f"{'─'*95}")

from scipy.stats import spearmanr

for p in range(n_pos):
    bias = "HIGH BIAS"   if train_std[p] < 1.3 else \
           "medium bias" if train_std[p] < 1.6 else \
           "low bias"
    row = (f"  {p+1}    " +
           "   ".join(f"{rank_matrix[v,p]}" for v in range(n_videos)) +
           f"     {train_avg[p]:.2f}      {train_std[p]:.2f}"
           f"       {test_rank[p]}     {shift[p]:+.2f}"
           f"       {count_ratio[p]:.2f}      {bias}")
    print(row)
print(f"{'─'*95}")

# Spearman correlation: does training std predict count ratio error?
ratio_error = np.abs(count_ratio - 1.0)   # how far from perfect prediction
rho_std,   p_std   = spearmanr(train_std,  ratio_error)
rho_shift, p_shift = spearmanr(np.abs(shift), ratio_error)

print(f"\n  Spearman correlation (train std vs |count ratio - 1|): "
      f"ρ={rho_std:.3f}  p={p_std:.3f}"
      f"  {'← bias explains errors' if p_std < 0.05 else '← not significant'}")
print(f"  Spearman correlation (|shift| vs |count ratio - 1|):  "
      f"ρ={rho_shift:.3f}  p={p_shift:.3f}"
      f"  {'← shift explains errors' if p_shift < 0.05 else '← not significant'}")

# Urgency rank in test video vs predicted count
urgency_rank = test_rank.astype(float)
rho_urg, p_urg = spearmanr(urgency_rank, count_ratio)
print(f"  Spearman correlation (test urgency rank vs count ratio): "
      f"ρ={rho_urg:.3f}  p={p_urg:.3f}"
      f"  {'← model tracks urgency' if p_urg < 0.05 else '← model does NOT track urgency'}")

print(f"\n  Positions with HIGH positional bias (train std < 1.3): "
      f"{[p+1 for p in range(n_pos) if train_std[p] < 1.3]}")
print(f"  Positions with LOW  positional bias (train std > 1.6): "
      f"{[p+1 for p in range(n_pos) if train_std[p] > 1.6]}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list("bw_rank", ["#3b82f6", "#f8fafc", "#ef4444"], N=256)

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5),
                         gridspec_kw={"width_ratios": [3, 1.4, 1.4]})
fig.suptitle("Bandwidth rank per dial position — layout bias analysis",
             fontsize=13, fontweight="bold")

# ── Left: heatmap ─────────────────────────────────────────────────────────────
ax = axes[0]
im = ax.imshow(rank_matrix.T, aspect="auto", cmap=cmap, vmin=1, vmax=6,
               interpolation="nearest")

# Annotate cells
for v in range(n_videos):
    for p in range(n_pos):
        r = rank_matrix[v, p]
        color = "white" if r in (1, 6) else "black"
        ax.text(v, p, str(r), ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)

# Column labels + highlights
video_labels = [f"V{v+1}\n(train)" for v in range(5)] + ["V6\n(val)", "V7\n(test)"]
ax.set_xticks(range(n_videos))
ax.set_xticklabels(video_labels, fontsize=9)
ax.set_yticks(range(n_pos))
ax.set_yticklabels([f"Position {p+1}" for p in range(n_pos)], fontsize=10)
ax.set_xlabel("Video", fontsize=11)
ax.set_ylabel("Dial position", fontsize=11)
ax.set_title("Bandwidth rank (1=lowest, 6=highest) per position per video", fontsize=10)

# Highlight test column
for p in range(n_pos):
    rect = plt.Rectangle((5.5, p - 0.5), 1, 1,
                          linewidth=2.5, edgecolor="black", facecolor="none")
    ax.add_patch(rect)

# Add training average ± std as text on right edge
for p in range(n_pos):
    ax.text(n_videos - 0.1, p,
            f"  avg={train_avg[p]:.1f} ±{train_std[p]:.2f}",
            va="center", fontsize=8, color="#555555")

plt.colorbar(im, ax=ax, label="Bandwidth rank", fraction=0.03, pad=0.01)

# ── Right: conflict bar chart ─────────────────────────────────────────────────
# The problem is not shift magnitude per se, but when the positional prior
# says "attend here" (high train avg) while the test signal says "low urgency"
# (low test rank). Positive shifts are harmless: signal and prior agree.
ax2 = axes[1]

# Color only negative shifts (prior > signal) as problematic
colors = ["#ef4444" if s < -1.5 else "#f97316" if s < 0 else
          "#d1fae5" for s in shift]
bars = ax2.barh(range(n_pos), shift, color=colors, edgecolor="white", height=0.6)
ax2.axvline(0, color="black", linewidth=1)
ax2.set_yticks(range(n_pos))
ax2.set_yticklabels([f"Pos {p+1}" for p in range(n_pos)], fontsize=10)
ax2.set_xlabel("Rank shift  (test − train avg)", fontsize=10)
ax2.set_title("Prior vs signal conflict\n(negative = prior over-estimates urgency)", fontsize=10)
ax2.set_xlim(-4, 4)
ax2.grid(axis="x", alpha=0.3)

for bar, s in zip(bars, shift):
    ax2.text(s + (0.15 if s >= 0 else -0.15), bar.get_y() + bar.get_height() / 2,
             f"{s:+.1f}", va="center", ha="left" if s >= 0 else "right",
             fontsize=10, fontweight="bold")

# Annotation explaining why positive shifts are harmless
ax2.text(0.98, 0.02,
         "Positive shifts: signal confirms\nor exceeds prior → no conflict",
         transform=ax2.transAxes, fontsize=7.5, ha="right", va="bottom",
         color="#555555", style="italic")

patches = [
    mpatches.Patch(color="#ef4444", label="Strong conflict: prior high, signal low → model over-predicts"),
    mpatches.Patch(color="#f97316", label="Mild conflict"),
    mpatches.Patch(color="#d1fae5", label="No conflict: signal ≥ prior → both agree"),
]
ax2.legend(handles=patches, fontsize=7.5, loc="upper left")

# ── Third panel: bias (train std) vs count ratio error ────────────────────────
ax3 = axes[2]
ratio_error = np.abs(count_ratio - 1.0)
scatter_colors = ["#ef4444" if train_std[p] < 1.3 else
                  "#f97316" if train_std[p] < 1.6 else
                  "#22c55e" for p in range(n_pos)]
ax3.scatter(train_std, ratio_error, c=scatter_colors, s=120, zorder=3, edgecolors="white")
for p in range(n_pos):
    ax3.annotate(f"D{p+1}", (train_std[p], ratio_error[p]),
                 textcoords="offset points", xytext=(6, 3), fontsize=9)

# Trend line
m, b = np.polyfit(train_std, ratio_error, 1)
x_line = np.linspace(train_std.min() - 0.1, train_std.max() + 0.1, 50)
ax3.plot(x_line, m * x_line + b, "k--", linewidth=1.2, alpha=0.6)

ax3.set_xlabel("Training rank std (lower = more biased)", fontsize=10)
ax3.set_ylabel("|Count ratio − 1|  (prediction error)", fontsize=10)
ax3.set_title(f"Does bias predict error?\nρ={rho_std:.2f}  p={p_std:.2f}", fontsize=10)
ax3.grid(alpha=0.3)

patches_bias = [
    mpatches.Patch(color="#ef4444", label="High bias (std < 1.3)"),
    mpatches.Patch(color="#f97316", label="Medium bias"),
    mpatches.Patch(color="#22c55e", label="Low bias (std > 1.6)"),
]
ax3.legend(handles=patches_bias, fontsize=7.5)

plt.tight_layout()
os.makedirs(config.RESULTS, exist_ok=True)
out_path = os.path.join(config.RESULTS, "layout_bias.png")
plt.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
