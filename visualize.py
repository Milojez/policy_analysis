"""
visualize.py
============
Autoregressively generates full scanpaths for all test participants and
compares the AOI count fraction distribution to the ground truth.

The model is seeded with the first PAST_FIXATIONS real fixations, then
predicts subsequent fixations one by one, feeding each prediction back
as context for the next step.

Plots saved to config.RESULTS:
  - recall_by_dial.png         — per-dial recall (single forward pass)
  - aoi_count_fractions.png    — human vs model AOI count fractions
                                  (like hm_ar_256_ur_dis_gur1_best_p1_T0.1_aoi_count_fractions)

Run:
    python visualize.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

import config
from dataset import PolicyDataset
from model import PolicyNet
from evaluate import run_evaluation
from build_dataset import (
    build_signal_window,
    load_signal_csv,
    denorm_saccade,
    denorm_duration,
)


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_best_model(device):
    ckpt = torch.load(config.CKPT, map_location=device)
    model = PolicyNet(
        use_fixations=config.USE_FIXATIONS,
        use_signal=config.USE_SIGNAL,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"Loaded: epoch={ckpt.get('epoch', '?')}  "
        f"best_val_acc={ckpt.get('best_val_acc', float('nan')) * 100:.2f}%"
    )
    return model


def load_payload():
    with open(config.DATA_PKL, "rb") as f:
        return pickle.load(f)


def load_test_dataset():
    """
    Load the stacked test .pt dataset.

    Expected config variable:
        config.TEST_PT
    """
    if not hasattr(config, "TEST_PT"):
        raise AttributeError(
            "config.TEST_PT is missing. Please add the path to your stacked "
            "test split .pt file in config.py"
        )

    if not os.path.exists(config.TEST_PT):
        raise FileNotFoundError(f"Test split not found: {config.TEST_PT}")

    return PolicyDataset(config.TEST_PT)


# ── Plot 1: per-dial recall ───────────────────────────────────────────────────

def plot_recall_by_dial(recall, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(recall))
    bars = ax.bar(x, recall * 100, color="steelblue", edgecolor="white")
    ax.axhline(
        100 / 6,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Random baseline ({100/6:.1f}%)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Dial {i+1}" for i in range(len(recall))])
    ax.set_ylabel("Recall (%)")
    ax.set_title(f"PolicyNet — Per-dial recall ({config.MODEL_NAME})\nTest video 7")
    ax.legend()
    ax.set_ylim(0, 100)

    for bar, r in zip(bars, recall):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{r*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Autoregressive scanpath generation ────────────────────────────────────────

def generate_scanpath(
    model,
    df_sig,
    time_end_arr,
    angle_cols,
    urgency_std,
    temporal_stats,
    t_end_video,
    device,
    max_fixations=500,
):
    """
    Autoregressively generate a full scanpath for one video.

    Starts at t = SIGNAL_LENGTH_S (first moment a full signal window exists).
    past_aois is all zeros — no real fixations are used as seed.
    Each predicted fixation becomes the new history (sliding window of length N).
    Stops when t_current reaches t_end_video or max_fixations is hit.

    Returns list of (aoi_1indexed, t_begin_s, t_end_s).
    """
    N = config.PAST_FIXATIONS
    t_current = config.SIGNAL_LENGTH_S
    past = [0] * N
    past_temp = [[0.0, 0.0]] * N   # [duration_norm, saccade_norm] per past fixation

    results = []

    while t_current < t_end_video and len(results) < max_fixations:
        window = build_signal_window(
            df_sig,
            time_end_arr,
            angle_cols,
            t_current,
            urgency_std,
        )

        pa_t  = torch.tensor([past],      dtype=torch.long,    device=device)
        pt_t  = torch.tensor([past_temp], dtype=torch.float32, device=device)
        sig_t = torch.tensor(window[None], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, temp_pred = model(pa_t, sig_t, pt_t)

        probs = torch.softmax(logits[0], dim=-1)
        aoi_0 = int(torch.multinomial(probs, 1).item())
        aoi_1 = aoi_0 + 1

        sacc_norm, dur_norm = temp_pred[0].cpu().tolist()
        saccade_s = denorm_saccade(sacc_norm, temporal_stats)
        duration_s = denorm_duration(dur_norm, temporal_stats)

        t_begin = t_current + saccade_s
        t_end = t_begin + duration_s

        results.append((aoi_1, t_begin, t_end))

        t_current = t_end
        if N > 0:
            past      = past[1:]      + [aoi_1]
            past_temp = past_temp[1:] + [[dur_norm, sacc_norm]]

    return results


# ── Plot 2: AOI count fractions ───────────────────────────────────────────────

def compute_aoi_count_fractions(model, gt_df, signal_data,
                                urgency_std, temporal_stats, device):
    """
    For each participant × test video:
      - Actual: count fixations that start after SIGNAL_LENGTH_S
      - Predicted: run full autoregressive generation from SIGNAL_LENGTH_S,
                   starting with zero past-fixation history, until video end

    Returns:
        actual_counts [6], pred_counts [6]
    """
    t_signal_start = config.SIGNAL_LENGTH_S
    test_pps = gt_df[gt_df["video"].isin(config.TEST_VIDEOS)]["pp"].unique()[:20]

    actual_counts = np.zeros(6, dtype=int)
    pred_counts = np.zeros(6, dtype=int)
    n_done = 0

    for pp in test_pps:
        for video in config.TEST_VIDEOS:
            if video not in signal_data:
                continue

            df_sig, time_end_arr, angle_cols = signal_data[video]
            t_end_video = float(time_end_arr[-1])

            grp = gt_df[(gt_df["pp"] == pp) & (gt_df["video"] == video)]
            grp = grp[grp["t_begin_s"] >= t_signal_start].sort_values("t_begin_s")

            for d in grp["dial"]:
                if 1 <= int(d) <= 6:
                    actual_counts[int(d) - 1] += 1

            generated = generate_scanpath(
                model,
                df_sig,
                time_end_arr,
                angle_cols,
                urgency_std,
                temporal_stats,
                t_end_video,
                device,
            )

            for aoi_1, _, _ in generated:
                if 1 <= int(aoi_1) <= 6:
                    pred_counts[int(aoi_1) - 1] += 1

            n_done += 1
            if n_done % 20 == 0:
                print(f"  {n_done} participant×video combinations done...")

    return actual_counts, pred_counts


def plot_aoi_count_fractions(actual_counts, pred_counts, out_path):
    dial_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"AOI fixation count fractions — {config.MODEL_NAME}\n"
        f"Autoregressive generation on test video 7 "
        f"(cold start from t={config.SIGNAL_LENGTH_S:.1f}s, "
        f"history window N={config.PAST_FIXATIONS})",
        fontsize=11,
    )

    for ax, counts, title in zip(
        axes,
        [actual_counts, pred_counts],
        ["Human (ground truth)", "PolicyNet (autoregressive)"],
    ):
        total = counts.sum()
        fracs = counts / max(total, 1)
        bottom = 0.0

        for d in range(6):
            ax.bar(
                0,
                fracs[d],
                bottom=bottom,
                color=dial_colors[d],
                label=f"Dial {d+1} ({fracs[d]*100:.1f}%)",
            )
            ax.text(
                0,
                bottom + fracs[d] / 2,
                f"D{d+1}\n{fracs[d]*100:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
            bottom += fracs[d]

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel("Fraction of fixations")
        ax.set_title(f"{title}\n({int(total):,} fixations)")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_best_model(device)
    payload = load_payload()

    urgency_std = payload["urgency_std"]
    temporal_stats = payload["temporal_stats"]

    os.makedirs(config.RESULTS, exist_ok=True)

    # Plot 1: recall by dial
    print("Loading stacked test dataset...")
    test_dataset = load_test_dataset()
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    _, recall, _, _, _ = run_evaluation(model, test_loader, device)
    plot_recall_by_dial(
        recall,
        os.path.join(config.RESULTS, "recall_by_dial.png"),
    )

    # Plot 2: autoregressive AOI count fractions
    print("\nLoading GT CSV and signal CSVs for autoregressive generation...")
    gt_df = pd.read_csv(config.GT_CSV)

    all_videos = sorted(gt_df["video"].unique())
    signal_data = {}

    for vnum in all_videos:
        loaded = load_signal_csv(vnum)

        if len(loaded) == 4:
            df_sig, _, _, angle_cols = loaded
        elif len(loaded) == 3:
            df_sig, _, angle_cols = loaded
        else:
            raise ValueError(
                f"Unexpected return format from load_signal_csv({vnum}). "
                f"Expected 3 or 4 values, got {len(loaded)}."
            )

        time_end_arr = df_sig["time_end"].values.astype(np.float64)
        signal_data[vnum] = (df_sig, time_end_arr, angle_cols)

    print("Generating scanpaths for all test participants...")
    actual_counts, pred_counts = compute_aoi_count_fractions(
        model,
        gt_df,
        signal_data,
        urgency_std,
        temporal_stats,
        device,
    )

    print(f"\nActual counts: {actual_counts}  (total {actual_counts.sum():,})")
    print(f"Pred counts:   {pred_counts}  (total {pred_counts.sum():,})")

    plot_aoi_count_fractions(
        actual_counts,
        pred_counts,
        os.path.join(config.RESULTS, "aoi_count_fractions.png"),
    )


if __name__ == "__main__":
    main()
