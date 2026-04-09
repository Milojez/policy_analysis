"""
explain_aoi_bias.py
===================
Per-participant urgency tracking analysis.

For each participant, computes the Spearman correlation between:
  - urgency rank of each dial (1=lowest, 6=highest bandwidth) in video 7
  - number of fixations that participant (or the model for that participant)
    made on each dial

This gives 83 correlation values for GT and 83 for the model,
with proper statistical power (n=83 participants, not n=6 dials).

Also produces the layout bias heatmap and count fraction comparison.

Run:
    python explain_aoi_bias.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, wilcoxon, ttest_rel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

TARGET_VIDEO = 7

# ── Layout config ──────────────────────────────────────────────────────────────
DIAL_CONFIG = np.array([
    [3, 6, 2, 4, 5, 1],   # video 1  (train)
    [4, 6, 1, 5, 2, 3],   # video 2  (train)
    [1, 3, 4, 5, 6, 2],   # video 3  (train)
    [5, 3, 2, 6, 1, 4],   # video 4  (train)
    [6, 2, 1, 3, 4, 5],   # video 5  (train)
    [3, 5, 2, 4, 1, 6],   # video 6  (val)
    [5, 1, 4, 3, 2, 6],   # video 7  (test)
], dtype=int)

n_videos, n_pos = DIAL_CONFIG.shape
train_idx = [0, 1, 2, 3, 4]

# rank_matrix[v, p] = bandwidth rank of position p+1 in video v+1
rank_matrix = np.zeros((n_videos, n_pos), dtype=int)
for v in range(n_videos):
    b   = np.argsort(DIAL_CONFIG[v]) + 1
    inv = np.zeros(n_pos + 1, dtype=int)
    for bw in range(1, n_pos + 1):
        inv[b[bw - 1]] = bw
    for p in range(n_pos):
        rank_matrix[v, p] = inv[p + 1]

train_avg = rank_matrix[train_idx].mean(axis=0)
train_std = rank_matrix[train_idx].std(axis=0)
test_rank = rank_matrix[-1].astype(float)   # urgency rank per dial in video 7

# ── Load data ──────────────────────────────────────────────────────────────────
pred_path = os.path.join(config.RESULTS, "predictions",
                         f"predicted_fixations_video{TARGET_VIDEO}.csv")
pred = pd.read_csv(pred_path)
gt   = pd.read_csv(config.GT_CSV)
gt   = gt[(gt["video"] == TARGET_VIDEO) & (gt["pp"].isin(pred["pp"].unique()))].copy()

pps   = sorted(pred["pp"].unique())
dials = [1, 2, 3, 4, 5, 6]

# ── Per-participant Spearman correlation ───────────────────────────────────────
# For each participant: correlate their fixation count per dial with urgency rank

gt_rhos   = []
pred_rhos = []

for pp in pps:
    gt_pp   = gt[gt["pp"]   == pp]
    pred_pp = pred[pred["pp"] == pp]

    gt_counts   = np.array([len(gt_pp[gt_pp["dial"]     == d]) for d in dials], dtype=float)
    pred_counts = np.array([len(pred_pp[pred_pp["dial"] == d]) for d in dials], dtype=float)

    # Only compute if participant has fixations on at least 3 dials
    if (gt_counts > 0).sum() >= 3:
        rho_gt, _   = spearmanr(test_rank, gt_counts)
        gt_rhos.append(rho_gt)

    if (pred_counts > 0).sum() >= 3:
        rho_pred, _ = spearmanr(test_rank, pred_counts)
        pred_rhos.append(rho_pred)

gt_rhos   = np.array(gt_rhos)
pred_rhos = np.array(pred_rhos)

# ── Statistical test ───────────────────────────────────────────────────────────
# Paired test on participants present in both
n_paired = min(len(gt_rhos), len(pred_rhos))
gt_paired   = gt_rhos[:n_paired]
pred_paired = pred_rhos[:n_paired]

stat_w, p_wilcoxon = wilcoxon(gt_paired, pred_paired)
stat_t, p_ttest    = ttest_rel(gt_paired, pred_paired)

# ── Print results ──────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  Per-participant urgency tracking (Spearman ρ per participant)")
print(f"  Video {TARGET_VIDEO}  |  urgency rank 1=lowest, 6=highest")
print(f"{'─'*65}")
print(f"  GT    — n={len(gt_rhos):3d}  "
      f"mean ρ={gt_rhos.mean():.3f}  "
      f"median ρ={np.median(gt_rhos):.3f}  "
      f"std={gt_rhos.std():.3f}  "
      f"% positive={100*(gt_rhos>0).mean():.1f}%")
print(f"  Model — n={len(pred_rhos):3d}  "
      f"mean ρ={pred_rhos.mean():.3f}  "
      f"median ρ={np.median(pred_rhos):.3f}  "
      f"std={pred_rhos.std():.3f}  "
      f"% positive={100*(pred_rhos>0).mean():.1f}%")
print(f"\n  Paired comparison (n={n_paired}):")
print(f"  Wilcoxon: stat={stat_w:.1f}  p={p_wilcoxon:.4f}"
      f"  {'significant' if p_wilcoxon < 0.05 else 'not significant'}")
print(f"  t-test:   t={stat_t:.3f}   p={p_ttest:.4f}"
      f"  {'significant' if p_ttest < 0.05 else 'not significant'}")
print(f"\n  Interpretation:")
if gt_rhos.mean() > 0.5:
    print(f"  GT participants strongly track urgency (mean ρ={gt_rhos.mean():.3f})")
if pred_rhos.mean() > 0.3:
    print(f"  Model also tracks urgency (mean ρ={pred_rhos.mean():.3f}) — "
          f"gap of {gt_rhos.mean()-pred_rhos.mean():.3f} to human level")
if p_wilcoxon < 0.05:
    print(f"  The gap is statistically significant (p={p_wilcoxon:.4f})")
else:
    print(f"  The gap is not statistically significant (p={p_wilcoxon:.4f})")
print(f"{'─'*65}\n")

# ── Aggregate count fractions ──────────────────────────────────────────────────
gt_counts_all   = np.array([len(gt[gt["dial"]     == d]) for d in dials], dtype=float)
pred_counts_all = np.array([len(pred[pred["dial"] == d]) for d in dials], dtype=float)
gt_frac   = gt_counts_all   / gt_counts_all.sum()   * 100
pred_frac = pred_counts_all / pred_counts_all.sum() * 100

# ── Figure ─────────────────────────────────────────────────────────────────────
cmap_rank = LinearSegmentedColormap.from_list(
    "bw_rank", ["#3b82f6", "#f8fafc", "#ef4444"], N=256)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(
    f"Urgency tracking analysis — {config.MODEL_NAME}  |  test video {TARGET_VIDEO}",
    fontsize=12, fontweight="bold")

dial_labels = [f"D{d}" for d in dials]
x = np.arange(n_pos)
w = 0.35

# ── Panel 1: Training layout diversity heatmap ────────────────────────────────
ax = axes[0]
show_idx    = train_idx + [-1]
show_labels = [f"V{v+1}\n(train)" for v in train_idx] + [f"V7\n(test)"]
im = ax.imshow(rank_matrix[show_idx].T, aspect="auto", cmap=cmap_rank,
               vmin=1, vmax=6, interpolation="nearest")
for vi, v in enumerate(show_idx):
    for p in range(n_pos):
        r = rank_matrix[v, p]
        ax.text(vi, p, str(r), ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if r in (1, 6) else "black")
for p in range(n_pos):
    ax.add_patch(plt.Rectangle((len(show_idx) - 1.5, p - 0.5), 1, 1,
                                lw=2.5, edgecolor="black", facecolor="none"))
ax.set_xticks(range(len(show_idx)))
ax.set_xticklabels(show_labels, fontsize=8)
ax.set_yticks(range(n_pos))
ax.set_yticklabels([f"D{p+1}  std={train_std[p]:.2f}" for p in range(n_pos)],
                   fontsize=9)
ax.set_title("Training layout diversity\n(std per position on y-axis)", fontsize=10)
plt.colorbar(im, ax=ax, label="BW rank", fraction=0.04, pad=0.02)

# ── Panel 2: Count fraction GT vs predicted ───────────────────────────────────
ax2 = axes[1]
ax2.bar(x - w/2, gt_frac,   w, label="GT",   color="steelblue", alpha=0.85)
ax2.bar(x + w/2, pred_frac, w, label="Pred", color="coral",     alpha=0.85)
for p in range(n_pos):
    diff = pred_frac[p] - gt_frac[p]
    ymax = max(gt_frac[p], pred_frac[p])
    color = "#ef4444" if diff > 1.5 else "#1d4ed8" if diff < -1.5 else "#888888"
    ax2.annotate(f"{diff:+.1f}%", xy=(p, ymax + 0.3),
                 ha="center", fontsize=9, fontweight="bold", color=color)
ax2.set_xticks(x); ax2.set_xticklabels(dial_labels)
ax2.set_ylabel("% of fixations"); ax2.set_ylim(0, 35)
ax2.set_title("Fixation count fraction\nGT vs predicted", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

# ── Panel 3: Per-participant ρ distributions ──────────────────────────────────
ax3 = axes[2]
bins = np.linspace(-1, 1, 25)
ax3.hist(gt_rhos,   bins=bins, alpha=0.6, color="steelblue",
         label=f"GT   mean={gt_rhos.mean():.3f}")
ax3.hist(pred_rhos, bins=bins, alpha=0.6, color="coral",
         label=f"Model mean={pred_rhos.mean():.3f}")
ax3.axvline(gt_rhos.mean(),   color="steelblue", linestyle="--", linewidth=2)
ax3.axvline(pred_rhos.mean(), color="coral",     linestyle="--", linewidth=2)
ax3.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)
ax3.set_xlabel("Spearman ρ (fixation count vs urgency rank)", fontsize=10)
ax3.set_ylabel("Number of participants", fontsize=10)
ax3.set_title(f"Per-participant urgency tracking\nn={len(gt_rhos)} participants",
              fontsize=10)
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

sig_str = f"p={p_wilcoxon:.3f} {'*' if p_wilcoxon<0.05 else 'ns'}"
ax3.text(0.05, 0.95, f"Wilcoxon {sig_str}",
         transform=ax3.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# ── Panel 4: Paired scatter GT ρ vs Model ρ per participant ──────────────────
ax4 = axes[3]
ax4.scatter(gt_paired, pred_paired, alpha=0.5, s=40,
            color="steelblue", edgecolors="white", linewidth=0.5)
lims = [-1, 1]
ax4.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect match")
ax4.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
ax4.axvline(0, color="gray", linewidth=0.8, alpha=0.4)
ax4.set_xlabel("GT Spearman ρ per participant", fontsize=10)
ax4.set_ylabel("Model Spearman ρ per participant", fontsize=10)
ax4.set_title("GT vs model urgency tracking\nper participant (paired)", fontsize=10)
ax4.set_xlim(-1, 1); ax4.set_ylim(-1, 1)
ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

# Quadrant annotation
ax4.text( 0.75,  0.05, "Model tracks\nurgency\nGT doesn't",
          fontsize=7, ha="center", color="#888888", transform=ax4.transAxes,
          va="bottom")
ax4.text( 0.05,  0.95, "Both track\nurgency",
          fontsize=7, ha="left", color="#22c55e", transform=ax4.transAxes,
          va="top")
ax4.text( 0.05,  0.05, "GT tracks\nurgency\nModel doesn't",
          fontsize=7, ha="left", color="#ef4444", transform=ax4.transAxes,
          va="bottom")

plt.tight_layout()
os.makedirs(config.RESULTS, exist_ok=True)
out_path = os.path.join(config.RESULTS, "urgency_tracking_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
