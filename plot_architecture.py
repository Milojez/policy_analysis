"""
Produces a clean architecture diagram for PolicyNet.
Run:  python plot_architecture.py
Saves: architecture_policy.png  (next to this script)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_policy.png")

# ── Colour palette ─────────────────────────────────────────────────────────────
C_INPUT  = "#dbeafe"   # light blue   — inputs
C_FIX    = "#dcfce7"   # light green  — fixation branch
C_SIG    = "#fef9c3"   # light yellow — signal branch
C_ATTN   = "#fce7f3"   # light pink   — cross-attention
C_FUSE   = "#e0e7ff"   # light indigo — fusion
C_HEAD   = "#ede9fe"   # light purple — output heads
C_OUT    = "#ffedd5"   # light orange — outputs
C_BORDER = "#374151"


def box(ax, x, y, w, h, label, sublabel=None, color="#ffffff", fontsize=9):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.025",
                          linewidth=1.2, edgecolor=C_BORDER,
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w / 2, cy + 0.10, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", zorder=4, color="#111827")
        ax.text(x + w / 2, cy - 0.13, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                zorder=4, color="#374151", style="italic")
    else:
        ax.text(x + w / 2, cy, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", zorder=4, color="#111827")


def arrow(ax, x0, y0, x1, y1, label=None, color="#374151", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=12,
                                linestyle=ls),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2 + 0.05, (y0 + y1) / 2
        ax.text(mx, my, label, fontsize=7, color="#6b7280", va="center", zorder=5)


def line(ax, x0, y0, x1, y1, color="#374151", ls="dashed"):
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.2,
            linestyle=ls, zorder=2)


fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 7)
ax.set_ylim(0, 18)
ax.axis("off")
fig.patch.set_facecolor("#f9fafb")

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(3.5, 17.65, "PolicyNet — Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#111827")
ax.text(3.5, 17.30, "policy_sepdur_h64_n2_str1_fbd05_hz10",
        ha="center", va="center", fontsize=9, color="#6b7280", style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# INPUTS
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 16.95, "INPUTS", fontsize=8, color="#6b7280", fontweight="bold")

box(ax, 0.15, 16.1, 2.5, 0.65,
    "past_aois  [B, 2]",
    "AOI labels: 0=pad, 1-6=dials", C_INPUT)
box(ax, 2.85, 16.1, 2.0, 0.65,
    "past_temporal  [B, 2, 2]",
    "duration + saccade per past fix.", C_INPUT)
box(ax, 5.05, 16.1, 1.8, 0.65,
    "signal  [B, 20, 6, 5]",
    "20 frames × 6 dials × 5 feats", C_INPUT)

# ══════════════════════════════════════════════════════════════════════════════
# FIXATION BRANCH  (left side)
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 15.7, "FIXATION BRANCH", fontsize=8, color="#166534", fontweight="bold")

box(ax, 0.15, 14.85, 1.15, 0.65,
    "aoi_emb",
    "Embed(7,16)", C_FIX)
box(ax, 1.45, 14.85, 1.35, 0.65,
    "temporal_proj",
    "Linear(2→16)", C_FIX)

arrow(ax, 1.40, 16.10, 0.72, 15.50)   # past_aois → aoi_emb
arrow(ax, 3.85, 16.10, 2.12, 15.50)   # past_temporal → temporal_proj

# cat + proj
box(ax, 0.15, 13.95, 2.65, 0.65,
    "cat + fix_input_proj",
    "cat([aoi_e, tmp_e]) → Linear(32→16)", C_FIX)
arrow(ax, 0.72, 14.85, 0.90, 14.60)
arrow(ax, 2.12, 14.85, 1.65, 14.60)

# FIX branch dropout note
ax.text(0.15, 13.78, "fix_branch_dropout=0.5 (training only)", fontsize=7,
        color="#6b7280", style="italic")

# GRU
box(ax, 0.15, 12.9, 2.65, 0.75,
    "GRU  (16→64, 1 layer)",
    "over N=2 past fixations → h_fix [B, 64]", C_FIX)
arrow(ax, 1.47, 13.95, 1.47, 13.65)

# h_fix output
box(ax, 0.55, 12.1, 1.85, 0.60, "h_fix  [B, 64]", color="#bbf7d0")
arrow(ax, 1.47, 12.90, 1.47, 12.70)

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL BRANCH  (right side)
# ══════════════════════════════════════════════════════════════════════════════
ax.text(3.60, 15.7, "SIGNAL BRANCH", fontsize=8, color="#854d0e", fontweight="bold")

box(ax, 3.60, 14.85, 0.90, 0.65, "dial_emb", "Embed(7,16)", C_SIG)
box(ax, 4.60, 14.85, 0.90, 0.65, "time_emb", "Embed(20,16)", C_SIG)

arrow(ax, 5.95, 16.10, 4.05, 15.50)   # signal → dial_emb
arrow(ax, 5.95, 16.10, 5.05, 15.50)   # signal → time_emb
arrow(ax, 5.95, 16.10, 5.95, 15.50)   # signal → cat directly

# cat + token_mlp
box(ax, 3.60, 13.80, 2.30, 0.80,
    "cat + token_mlp",
    "cat([sig,dial_e,time_e]) → Linear(37→64)\n→ LN → ReLU → Linear(64→64) → ReLU", C_SIG, fontsize=8)
arrow(ax, 4.05, 14.85, 4.30, 14.60)
arrow(ax, 5.05, 14.85, 4.75, 14.60)
arrow(ax, 5.95, 15.50, 5.10, 14.60)

ax.text(6.0, 14.20, "T = 120 tokens\n(20×6)", fontsize=7.5,
        color="#6b7280", va="center", style="italic")

# Transformer encoder
box(ax, 3.60, 12.85, 2.30, 0.75,
    "TransformerEncoder  ×1 block",
    "self-attn (4 heads) + FFN  →  tok_h [B, 120, 64]", C_SIG, fontsize=8)
arrow(ax, 4.75, 13.80, 4.75, 13.60)

# ══════════════════════════════════════════════════════════════════════════════
# DUAL CROSS-ATTENTION POOLING
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 12.55, "DUAL CROSS-ATTENTION POOLING", fontsize=8,
        color="#9d174d", fontweight="bold")

# Shared K/V
box(ax, 2.65, 11.90, 1.90, 0.65,
    "k_proj / v_proj",
    "Linear(64→64) shared", C_ATTN)
arrow(ax, 4.75, 12.85, 4.00, 12.55)   # tok_h → k/v

# Conditioned path
box(ax, 0.15, 11.00, 2.35, 0.75,
    "Conditioned cross-attn",
    "Q = q_proj(h_fix)  K/V = signal tokens\n4 heads  →  pooled_cond [B, 64]", C_ATTN, fontsize=8)
arrow(ax, 1.47, 12.10, 1.47, 11.75)   # h_fix → cond attn
arrow(ax, 2.65, 12.22, 1.70, 11.75)   # k/v → cond attn (dashed)

# Global path
box(ax, 2.65, 11.00, 2.35, 0.75,
    "Global cross-attn",
    "Q = global_query (learned param)\nK/V = signal tokens  →  pooled_glob [B, 64]", C_ATTN, fontsize=8)
arrow(ax, 3.82, 11.90, 3.82, 11.75)
# global_query annotation
ax.text(5.1, 11.38, "global_query\n(nn.Parameter)", fontsize=7.5,
        color="#9d174d", ha="center", style="italic")
arrow(ax, 5.0, 11.38, 4.95, 11.45, color="#9d174d")

# sig_fc
box(ax, 1.20, 10.10, 2.50, 0.70,
    "sig_fc",
    "cat([pooled_cond, pooled_glob]) → Linear(128→64) → ReLU", C_ATTN, fontsize=8)
arrow(ax, 1.32, 11.00, 2.10, 10.80)
arrow(ax, 3.82, 11.00, 2.95, 10.80)

box(ax, 1.60, 9.35, 1.70, 0.55, "h_sig  [B, 64]", color="#fce7f3")
arrow(ax, 2.45, 10.10, 2.45, 9.90)

# ══════════════════════════════════════════════════════════════════════════════
# FUSION
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 9.10, "FUSION", fontsize=8, color="#3730a3", fontweight="bold")

box(ax, 0.80, 8.30, 2.80, 0.65,
    "fusion",
    "cat([h_fix, h_sig]) → Linear(128→64) → ReLU  →  fused [B, 64]", C_FUSE, fontsize=8)
arrow(ax, 1.47, 9.35, 1.80, 8.95)
arrow(ax, 2.45, 9.35, 2.65, 8.95)

box(ax, 1.50, 7.60, 1.80, 0.55, "fused  [B, 64]", color="#c7d2fe")
arrow(ax, 2.20, 8.30, 2.20, 8.15)

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT HEADS
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.15, 7.35, "OUTPUT HEADS", fontsize=8, color="#4c1d95", fontweight="bold")

# Split to dial head and temporal head
arrow(ax, 2.20, 7.60, 1.20, 7.05)
arrow(ax, 2.20, 7.60, 3.80, 7.05)

# Dial head
box(ax, 0.15, 6.15, 2.00, 0.75,
    "dial_head",
    "Linear(64→6)\n→ logits [B, 6]", C_HEAD)
arrow(ax, 1.15, 7.05, 1.15, 6.90)

# Temporal head inputs
box(ax, 2.75, 6.50, 1.30, 0.55,
    "chosen_dial",
    "aoi_emb → [B, 16]", C_HEAD, fontsize=8)
box(ax, 4.20, 6.50, 2.65, 0.55,
    "future_signal  [B, 5, 6, 5]",
    "mean pool → future_sig_fc → [B, 64]", C_HEAD, fontsize=8)

arrow(ax, 3.80, 7.05, 3.40, 7.05)   # fused → temporal area
arrow(ax, 3.80, 7.05, 3.80, 7.05)

box(ax, 2.75, 5.40, 4.10, 0.85,
    "temporal_head",
    "cat([fused.detach(), chosen_e, fut_h])\n→ Linear(64+16+64→4)\n→ [sacc_μ, dur_μ, sacc_logσ, dur_logσ]",
    C_HEAD, fontsize=8)
arrow(ax, 3.40, 7.60, 3.40, 6.25,  color="#374151")   # fused.detach() → temporal
arrow(ax, 3.40, 6.50, 3.60, 6.25)
arrow(ax, 5.52, 6.50, 5.20, 6.25)

ax.text(2.60, 5.95, "fused\n.detach()", fontsize=7, color="#9d174d",
        ha="center", style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.15, 4.50, 2.00, 0.65,
    "logits  [B, 6]",
    "next dial probabilities", C_OUT)
box(ax, 2.75, 4.50, 4.10, 0.65,
    "temporal  [B, 4]",
    "sacc_μ, dur_μ, sacc_logσ, dur_logσ  (NLL loss)", C_OUT)

arrow(ax, 1.15, 6.15, 1.15, 5.15)
arrow(ax, 4.80, 5.40, 4.80, 5.15)

# Loss annotations
ax.text(1.15, 4.25, "CrossEntropy loss", fontsize=7.5,
        ha="center", color="#6b7280", style="italic")
ax.text(4.80, 4.25, "Gaussian NLL loss  (λ=0.05)", fontsize=7.5,
        ha="center", color="#6b7280", style="italic")

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT, edgecolor=C_BORDER, label="Input"),
    mpatches.Patch(facecolor=C_FIX,   edgecolor=C_BORDER, label="Fixation branch"),
    mpatches.Patch(facecolor=C_SIG,   edgecolor=C_BORDER, label="Signal branch"),
    mpatches.Patch(facecolor=C_ATTN,  edgecolor=C_BORDER, label="Cross-attention pooling"),
    mpatches.Patch(facecolor=C_FUSE,  edgecolor=C_BORDER, label="Fusion"),
    mpatches.Patch(facecolor=C_HEAD,  edgecolor=C_BORDER, label="Output heads"),
    mpatches.Patch(facecolor=C_OUT,   edgecolor=C_BORDER, label="Output"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          framealpha=0.9, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
plt.show()
