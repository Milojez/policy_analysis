"""
generate_predictions.py
=======================
Autoregressively generates predicted fixation sequences for a target video
using the trained PolicyNet, producing a CSV that mirrors the GT format:

    pp, video, dial, t_begin_s, t_mid_s, t_end_s, duration_s

One scanpath is generated per participant found in the GT CSV for the target
video. Because prediction uses torch.multinomial (stochastic), each participant
gets a different sample from the same model distribution.

Run:
    python generate_predictions.py
"""

import os
import sys
import csv
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

import config
from model import PolicyNet
from build_dataset import (
    build_signal_window,
    load_signal_csv,
    denorm_saccade,
    denorm_duration,
)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_best_model(device):
    ckpt = torch.load(config.CKPT, map_location=device)
    model = PolicyNet(
        use_fixations=config.USE_FIXATIONS,
        use_signal=config.USE_SIGNAL,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}  "
        f"best_val_acc={ckpt.get('best_val_acc', float('nan')) * 100:.2f}%"
    )
    return model


# ── Autoregressive generation (one scanpath per participant) ───────────────────

def generate_scanpath(
    model,
    df_sig,
    time_end_arr,
    angle_cols,
    speed_cols,
    urgency_std,
    speed_std,
    temporal_stats,
    t_end_video,
    device,
    max_fixations=1000,
):
    """
    Generate one full scanpath autoregressively for a single participant/video.

    Returns list of dicts with keys:
        dial, t_begin_s, t_mid_s, t_end_s, duration_s
    """
    N = config.PAST_FIXATIONS
    t_current = config.SIGNAL_LENGTH_S
    past      = [0] * N                   # 0 = pad (no prior fixation)
    past_temp = [[0.0, 0.0]] * N          # [duration_norm, saccade_norm]

    rows = []

    while t_current < t_end_video and len(rows) < max_fixations:
        window = build_signal_window(
            df_sig,
            time_end_arr,
            angle_cols,
            speed_cols,
            t_current,
            urgency_std,
            speed_std,
        )

        pa_t  = torch.tensor([past],       dtype=torch.long,    device=device)
        pt_t  = torch.tensor([past_temp],  dtype=torch.float32, device=device)
        sig_t = torch.tensor(window[None], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, temp_pred = model(pa_t, sig_t, pt_t)

        probs  = torch.softmax(logits[0], dim=-1)
        aoi_0  = int(torch.multinomial(probs, 1).item())
        aoi_1  = aoi_0 + 1                # 1-indexed dial position

        sacc_norm, dur_norm = temp_pred[0].cpu().tolist()
        saccade_s  = denorm_saccade(sacc_norm, temporal_stats)
        duration_s = denorm_duration(dur_norm, temporal_stats)

        t_begin = t_current + saccade_s
        t_end   = t_begin + duration_s
        t_mid   = (t_begin + t_end) / 2.0

        rows.append({
            "dial":       aoi_1,
            "t_begin_s":  round(t_begin,   4),
            "t_mid_s":    round(t_mid,     4),
            "t_end_s":    round(t_end,     4),
            "duration_s": round(duration_s, 4),
        })

        t_current = t_end
        if N > 0:
            past      = past[1:]      + [aoi_1]
            past_temp = past_temp[1:] + [[dur_norm, sacc_norm]]

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

TARGET_VIDEO = 7   # change this to generate for a different video
SEED         = None  # set to an int for reproducible output, e.g. 42


def main():
    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_best_model(device)

    # Load normalization stats from payload
    with open(config.DATA_PKL, "rb") as f:
        payload = pickle.load(f)

    urgency_std    = payload["urgency_std"]
    speed_std      = payload["speed_std"]
    temporal_stats = payload["temporal_stats"]

    # Load signal CSV for the target video
    print(f"\nLoading signal CSV for video {TARGET_VIDEO}...")
    loaded = load_signal_csv(TARGET_VIDEO)
    if len(loaded) == 5:
        df_sig, _, _, angle_cols, speed_cols = loaded
    elif len(loaded) == 4:
        df_sig, _, _, angle_cols = loaded
        speed_cols = {}
    else:
        raise ValueError(f"Unexpected return from load_signal_csv: {len(loaded)} values")

    time_end_arr = df_sig["time_end"].values.astype(np.float64)
    t_end_video  = float(time_end_arr[-1])
    print(f"  Signal duration: {t_end_video:.1f} s  ({len(time_end_arr):,} frames)")

    # Get participant list from GT CSV for this video
    gt_df = pd.read_csv(config.GT_CSV)
    pps   = sorted(gt_df[gt_df["video"] == TARGET_VIDEO]["pp"].unique())
    print(f"  Participants in GT for video {TARGET_VIDEO}: {len(pps)}")

    # Generate one scanpath per participant
    os.makedirs(config.RESULTS, exist_ok=True)
    out_path = os.path.join(
        config.RESULTS,
        f"predicted_fixations_video{TARGET_VIDEO}.csv",
    )

    fieldnames = ["pp", "video", "dial", "t_begin_s", "t_mid_s", "t_end_s", "duration_s"]

    all_durations = []
    all_saccades  = []
    total_fixations = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, pp in enumerate(pps):
            rows = generate_scanpath(
                model,
                df_sig,
                time_end_arr,
                angle_cols,
                speed_cols,
                urgency_std,
                speed_std,
                temporal_stats,
                t_end_video,
                device,
            )

            for row in rows:
                writer.writerow({"pp": pp, "video": TARGET_VIDEO, **row})
                all_durations.append(row["duration_s"])
                # saccade = gap between previous fixation end and this fixation begin
                # approximated as t_begin_s - t_end_s of previous row (within same pp)

            # collect saccade lengths within this participant's scanpath
            for j in range(1, len(rows)):
                saccade = rows[j]["t_begin_s"] - rows[j - 1]["t_end_s"]
                all_saccades.append(saccade)

            total_fixations += len(rows)

            if (i + 1) % 10 == 0 or (i + 1) == len(pps):
                print(f"  [{i+1:3d}/{len(pps)}] pp={pp}  "
                      f"fixations generated so far: {total_fixations:,}")

    all_durations = np.array(all_durations)
    all_saccades  = np.array(all_saccades)

    print(f"\nDone. {total_fixations:,} predicted fixations for {len(pps)} participants.")

    print(f"\n{'─'*55}")
    print(f"  Predicted (video {TARGET_VIDEO})")
    print(f"{'─'*55}")
    print(f"  Fixation duration (s):  mean={all_durations.mean():.3f}  std={all_durations.std():.3f}"
          f"  min={all_durations.min():.3f}  max={all_durations.max():.3f}")
    print(f"  Saccade length   (s):  mean={all_saccades.mean():.3f}  std={all_saccades.std():.3f}"
          f"  min={all_saccades.min():.3f}  max={all_saccades.max():.3f}")

    # Ground-truth stats for the same video and participants
    gt_video = gt_df[gt_df["video"] == TARGET_VIDEO].copy()
    gt_dur = gt_video["duration_s"].values.astype(float)

    gt_sacc = []
    for pp in pps:
        pp_rows = gt_video[gt_video["pp"] == pp].sort_values("t_begin_s")
        t_begins = pp_rows["t_begin_s"].values
        t_ends   = pp_rows["t_end_s"].values
        for j in range(1, len(t_begins)):
            gt_sacc.append(t_begins[j] - t_ends[j - 1])
    gt_sacc = np.array(gt_sacc)

    print(f"\n{'─'*55}")
    print(f"  Ground truth (video {TARGET_VIDEO})")
    print(f"{'─'*55}")
    print(f"  Fixation duration (s):  mean={gt_dur.mean():.3f}  std={gt_dur.std():.3f}"
          f"  min={gt_dur.min():.3f}  max={gt_dur.max():.3f}")
    if len(gt_sacc) > 0:
        print(f"  Saccade length   (s):  mean={gt_sacc.mean():.3f}  std={gt_sacc.std():.3f}"
              f"  min={gt_sacc.min():.3f}  max={gt_sacc.max():.3f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
