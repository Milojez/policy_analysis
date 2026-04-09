"""
visualize.py
============
Autoregressively generates full scanpaths for test participants and compares
against ground truth. Produces 4 plots saved to config.RESULTS:

  recall_by_dial.png          — per-dial recall (single forward pass on test set)
  aoi_count_fractions.png     — fixation count fraction: human vs model
  aoi_time_fractions.png      — time-on-dial fraction: human vs model
  duration_distribution.png   — fixation duration distribution: human vs model

Run:
    python visualize.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

import config
from dataset import PolicyDataset
from model import PolicyNet
from evaluate import run_evaluation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "datasets_built"))
from build_dataset import (
    build_signal_window,
    build_future_signal_window,
    load_signal_csv,
    denorm_saccade,
    denorm_duration,
)

# ── Run parameters ─────────────────────────────────────────────────────────────
# Number of GT participants to generate predictions for.
# Each participant produces one full autoregressive scanpath (~90 s).
# Lower = faster run; set to None to use all participants in the GT CSV.
N_PARTICIPANTS = 20


# ── Load helpers ───────────────────────────────────────────────────────────────

def load_best_model(device):
    ckpt = torch.load(config.CKPT, map_location=device)
    model = PolicyNet(use_fixations=config.USE_FIXATIONS,
                      use_signal=config.USE_SIGNAL).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: epoch={ckpt.get('epoch', '?')}  "
          f"best_val_acc={ckpt.get('best_val_acc', float('nan')) * 100:.2f}%")
    return model


# ── Autoregressive scanpath generation ────────────────────────────────────────

def generate_scanpath(model, df_sig, time_end_arr, angle_cols, speed_cols,
                      urgency_std, speed_std, temporal_stats, t_end_video,
                      device, max_fixations=500):
    N         = config.PAST_FIXATIONS
    t_current = config.SIGNAL_LENGTH_S
    past      = [0] * N
    past_temp = [[0.0, 0.0]] * N

    results = []
    while t_current < t_end_video and len(results) < max_fixations:
        window = build_signal_window(df_sig, time_end_arr, angle_cols, speed_cols,
                                     t_current, urgency_std, speed_std)
        pa_t  = torch.tensor([past],       dtype=torch.long,    device=device)
        pt_t  = torch.tensor([past_temp],  dtype=torch.float32, device=device)
        sig_t = torch.tensor(window[None], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, _ = model(pa_t, sig_t, pt_t)

        probs = torch.softmax(logits[0], dim=-1)
        aoi_0 = int(torch.multinomial(probs, 1).item())
        aoi_1 = aoi_0 + 1

        future_window = build_future_signal_window(
            df_sig, time_end_arr, angle_cols, speed_cols,
            t_current, urgency_std, speed_std)
        fut_t  = torch.tensor(future_window[None], dtype=torch.float32, device=device)
        dial_t = torch.tensor([aoi_0], dtype=torch.long, device=device)

        with torch.no_grad():
            _, temp_pred = model(pa_t, sig_t, pt_t,
                                 chosen_dial=dial_t, future_signal=fut_t)

        mu    = temp_pred[0, :2]
        sigma = temp_pred[0, 2:].exp().clamp(min=1e-4)
        sacc_norm, dur_norm = torch.normal(mu, sigma).cpu().tolist()
        saccade_s  = denorm_saccade(sacc_norm, temporal_stats)
        duration_s = denorm_duration(dur_norm, temporal_stats)

        t_begin = t_current + saccade_s
        t_end   = t_begin + duration_s
        results.append((aoi_1, t_begin, t_end))

        t_current = t_end
        if N > 0:
            past      = past[1:]      + [aoi_1]
            past_temp = past_temp[1:] + [[dur_norm, sacc_norm]]

    return results


# ── Compute stats ──────────────────────────────────────────────────────────────

def compute_aoi_stats(model, gt_df, signal_data, urgency_std, speed_std,
                      temporal_stats, device):
    t_signal_start = config.SIGNAL_LENGTH_S
    all_pps = gt_df[gt_df["video"].isin(config.TEST_VIDEOS)]["pp"].unique()
    test_pps = all_pps if N_PARTICIPANTS is None else all_pps[:N_PARTICIPANTS]
    print(f"  Running on {len(test_pps)} participants "
          f"({'all' if N_PARTICIPANTS is None else f'N_PARTICIPANTS={N_PARTICIPANTS}'})")

    actual_counts = np.zeros(6, dtype=int)
    pred_counts   = np.zeros(6, dtype=int)
    actual_times  = np.zeros(6, dtype=float)
    pred_times    = np.zeros(6, dtype=float)
    actual_durations = []
    pred_durations   = []

    for n_done, pp in enumerate(test_pps):
        for video in config.TEST_VIDEOS:
            if video not in signal_data:
                continue
            df_sig, time_end_arr, angle_cols, speed_cols = signal_data[video]
            t_end_video = float(time_end_arr[-1])

            grp = gt_df[(gt_df["pp"] == pp) & (gt_df["video"] == video)]
            grp = grp[grp["t_begin_s"] >= t_signal_start].sort_values("t_begin_s")

            for _, row in grp.iterrows():
                d = int(row["dial"])
                if 1 <= d <= 6:
                    actual_counts[d - 1] += 1
                    actual_times[d - 1]  += float(row["duration_s"])
                    actual_durations.append(float(row["duration_s"]))

            generated = generate_scanpath(model, df_sig, time_end_arr,
                                          angle_cols, speed_cols,
                                          urgency_std, speed_std,
                                          temporal_stats, t_end_video, device)
            for aoi_1, t_begin, t_end in generated:
                if 1 <= int(aoi_1) <= 6:
                    dur = max(t_end - t_begin, 0.0)
                    pred_counts[int(aoi_1) - 1] += 1
                    pred_times[int(aoi_1) - 1]  += dur
                    pred_durations.append(dur)

        if (n_done + 1) % 10 == 0:
            print(f"  {n_done + 1}/{len(test_pps)} participants done...")

    return (actual_counts, pred_counts, actual_times, pred_times,
            np.array(actual_durations), np.array(pred_durations))


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_recall_by_dial(recall, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x    = np.arange(len(recall))
    bars = ax.bar(x, recall * 100, color="steelblue", edgecolor="white")
    ax.axhline(100 / 6, color="red", linestyle="--", linewidth=1.2,
               label=f"Random baseline ({100/6:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Dial {i+1}" for i in range(len(recall))])
    ax.set_ylabel("Recall (%)")
    ax.set_title(f"PolicyNet — Per-dial recall ({config.MODEL_NAME})\nTest videos {config.TEST_VIDEOS}")
    ax.legend()
    ax.set_ylim(0, 100)
    for bar, r in zip(bars, recall):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{r*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _stacked_bar(ax, values, dial_colors, ylabel, title, total_label):
    total  = values.sum()
    fracs  = values / max(total, 1e-9)
    bottom = 0.0
    for d in range(6):
        ax.bar(0, fracs[d], bottom=bottom, color=dial_colors[d],
               label=f"Dial {d+1} ({fracs[d]*100:.1f}%)")
        if fracs[d] > 0.03:
            ax.text(0, bottom + fracs[d] / 2, f"D{d+1}\n{fracs[d]*100:.0f}%",
                    ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        bottom += fracs[d]
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({total_label})")
    ax.legend(loc="upper right", fontsize=8)


def _subtitle():
    return (f"Autoregressive generation — test videos {config.TEST_VIDEOS} "
            f"| N={len(config.TEST_VIDEOS)*N_PARTICIPANTS if N_PARTICIPANTS else 'all'} scanpaths "
            f"| cold start t={config.SIGNAL_LENGTH_S:.1f}s")


def plot_aoi_count_fractions(actual_counts, pred_counts, out_path):
    dial_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"AOI fixation count fractions — {config.MODEL_NAME}\n{_subtitle()}", fontsize=11)
    _stacked_bar(axes[0], actual_counts, dial_colors, "Fraction of fixations",
                 "Human (ground truth)", f"{int(actual_counts.sum()):,} fixations")
    _stacked_bar(axes[1], pred_counts, dial_colors, "Fraction of fixations",
                 "PolicyNet (autoregressive)", f"{int(pred_counts.sum()):,} fixations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_aoi_time_fractions(actual_times, pred_times, out_path):
    dial_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"AOI time-on-dial fractions — {config.MODEL_NAME}\n{_subtitle()}", fontsize=11)
    _stacked_bar(axes[0], actual_times, dial_colors, "Fraction of total fixation time",
                 "Human (ground truth)", f"{actual_times.sum():.1f} s total")
    _stacked_bar(axes[1], pred_times, dial_colors, "Fraction of total fixation time",
                 "PolicyNet (autoregressive)", f"{pred_times.sum():.1f} s total")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_duration_distribution(actual_durations, pred_durations, out_path):
    clip = 2.0
    bins = np.linspace(0, clip, 60)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Fixation duration distribution — {config.MODEL_NAME}\n{_subtitle()}", fontsize=11)
    for ax, durations, label, color in [
        (axes[0], actual_durations, "Human (GT)",  "steelblue"),
        (axes[1], pred_durations,   "PolicyNet",   "coral"),
    ]:
        d = durations[durations <= clip]
        ax.hist(d, bins=bins, color=color, edgecolor="white", linewidth=0.4, density=True)
        ax.axvline(np.median(durations), color="black", linestyle="--", linewidth=1.2,
                   label=f"Median {np.median(durations):.3f} s")
        ax.axvline(np.mean(durations), color="black", linestyle=":", linewidth=1.2,
                   label=f"Mean {np.mean(durations):.3f} s")
        ax.set_xlabel("Fixation duration (s)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}  (n={len(durations):,}  std={np.std(durations):.3f}s)")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_best_model(device)

    with open(config.DATA_PKL, "rb") as f:
        payload = pickle.load(f)
    urgency_std    = payload["urgency_std"]
    speed_std      = payload["speed_std"]
    temporal_stats = payload["temporal_stats"]

    os.makedirs(config.RESULTS, exist_ok=True)

    # ── Recall by dial (single forward pass on test set) ──────────────────────
    print("Loading test dataset...")
    test_loader = DataLoader(PolicyDataset(config.TEST_PT),
                             batch_size=512, shuffle=False, num_workers=0)
    _, recall, _, _, _ = run_evaluation(model, test_loader, device)
    plot_recall_by_dial(recall, os.path.join(config.RESULTS, "recall_by_dial.png"))

    # ── Autoregressive generation ─────────────────────────────────────────────
    print("\nLoading GT CSV and signal CSVs...")
    gt_df = pd.read_csv(config.GT_CSV)
    signal_data = {}
    for vnum in sorted(gt_df["video"].unique()):
        loaded = load_signal_csv(vnum)
        df_sig, _, _, angle_cols, speed_cols = loaded if len(loaded) == 5 else (*loaded, {})
        signal_data[vnum] = (df_sig, df_sig["time_end"].values.astype(np.float64),
                             angle_cols, speed_cols)

    print(f"Generating scanpaths...")
    (actual_counts, pred_counts, actual_times, pred_times,
     actual_durations, pred_durations) = compute_aoi_stats(
        model, gt_df, signal_data, urgency_std, speed_std, temporal_stats, device)

    print(f"\nActual counts: {actual_counts}  total={actual_counts.sum():,}")
    print(f"Pred counts:   {pred_counts}  total={pred_counts.sum():,}")
    print(f"Actual times:  {np.round(actual_times, 1)}  total={actual_times.sum():.1f}s")
    print(f"Pred times:    {np.round(pred_times, 1)}  total={pred_times.sum():.1f}s")

    plot_aoi_count_fractions(actual_counts, pred_counts,
                             os.path.join(config.RESULTS, "aoi_count_fractions.png"))
    plot_aoi_time_fractions(actual_times, pred_times,
                            os.path.join(config.RESULTS, "aoi_time_fractions.png"))
    plot_duration_distribution(actual_durations, pred_durations,
                               os.path.join(config.RESULTS, "duration_distribution.png"))


if __name__ == "__main__":
    main()
