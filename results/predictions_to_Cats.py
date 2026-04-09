# Translates the model predictions CSV to a CatS frame-level CSV,
# mirroring the logic of fixations_to_Cats.py.
#
# Input:  config.RESULTS/predicted_fixations_video{TARGET_VIDEO}.csv
# Output: config.RESULTS/CatS_from_predicted_fixations_video{TARGET_VIDEO}.csv

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

TARGET_VIDEO = 7   # must match the video used in generate_predictions.py

FPS         = 50
VIDEO_LEN_S = 90
NFRAMES     = VIDEO_LEN_S * FPS   # 4500

# ── Load predictions ───────────────────────────────────────────────────────────

in_path = os.path.join(
    config.RESULTS,
    f"predicted_fixations_video{TARGET_VIDEO}.csv",
)
fix = pd.read_csv(in_path)
print(f"Loaded {len(fix):,} predicted fixations from:\n  {in_path}")

fix = fix.sort_values(["pp", "video", "t_begin_s", "t_end_s"]).reset_index(drop=True)

pps    = sorted(fix["pp"].unique())
groups = {k: g for k, g in fix.groupby(["pp", "video"], sort=False)}

print(f"Participants: {len(pps)}  Videos: {[TARGET_VIDEO]}")

# ── Frame mapping ──────────────────────────────────────────────────────────────

out_rows = []
n_clamped = 0

for pp in pps:
    frames = np.full(NFRAMES, np.nan)

    g = groups.get((pp, TARGET_VIDEO))
    if g is not None:
        for _, r in g.iterrows():
            tb   = float(r["t_begin_s"])
            te   = float(r["t_end_s"])
            dial = int(r["dial"])

            k_start = int(np.floor(tb * FPS)) + 1
            k_end   = int(np.ceil(te * FPS))

            # Clamp to valid frame range (model temporal predictions may
            # slightly overshoot video end; raises would be too strict here)
            k_start = max(1, min(k_start, NFRAMES))
            k_end   = max(1, min(k_end,   NFRAMES))

            if k_start > k_end:
                continue   # zero-length interval after clamping; skip

            if k_start != int(np.floor(tb * FPS)) + 1 or k_end != int(np.ceil(te * FPS)):
                n_clamped += 1

            frames[k_start - 1:k_end] = dial

    out_rows.append(pd.DataFrame({
        "pp":    pp,
        "video": TARGET_VIDEO,
        "frame": np.arange(1, NFRAMES + 1),
        "dial":  frames,
    }))

result = pd.concat(out_rows, ignore_index=True)
result["dial"] = result["dial"].astype("Int64")
result = result.sort_values(["pp", "video", "frame"]).reset_index(drop=True)

if n_clamped > 0:
    print(f"  Note: {n_clamped} fixation intervals were clamped to [1, {NFRAMES}]")

# ── Save ───────────────────────────────────────────────────────────────────────

predictions_dir = os.path.join(config.RESULTS, "predictions")
os.makedirs(predictions_dir, exist_ok=True)

out_path = os.path.join(
    predictions_dir,
    f"CatS_from_predicted_fixations_video{TARGET_VIDEO}.csv",
)
result.to_csv(out_path, index=False, na_rep="nan")
print(f"Wrote {len(result):,} rows → {out_path}")
