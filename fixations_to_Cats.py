#This script should translate the csv with fixations to CatS csv
import pandas as pd
import numpy as np
import warnings
FPS = 50
VIDEO_LEN_S = 90
NFRAMES = VIDEO_LEN_S * FPS  # 4500

NRPP = 92
VIDEOS = range(1, 8)         # 1..7
PPS = range(1, NRPP + 1)     # 1..92

# Input CSV columns:
# pp,video,dial,t_begin_s,t_mid_s,t_end_s,duration_s,x_fix,y_fix
fix = pd.read_csv("../../all/all.mat_fix_processed/GT_all_fixation_noDial7.csv")

# Deterministic overwrite order
fix = fix.sort_values(["pp", "video", "t_begin_s", "t_end_s"]).reset_index(drop=True)

# Group for quick lookup
groups = {k: g for k, g in fix.groupby(["pp", "video"], sort=False)}

out_rows = []

for pp in PPS:
    for video in VIDEOS:
        frames = np.full(NFRAMES, np.nan)  # NaN everywhere by default

        g = groups.get((pp, video))
        if g is not None:
            for _, r in g.iterrows():
                tb = float(r["t_begin_s"])
                te = float(r["t_end_s"])
                dial = int(r["dial"])

                # Map fixation interval to overlapping frames (1-based)
                k_start = int(np.floor(tb * FPS)) + 1
                k_end   = int(np.ceil(te * FPS))

                # --- sanity checks ---
                # --- hard range checks ---
                if k_start < 1 or k_start > NFRAMES:
                    raise ValueError(
                        f"k_start={k_start} outside [1,{NFRAMES}] "
                        f"(pp={pp}, video={video}, t_begin_s={tb})"
                    )

                if k_end < 1 or k_end > NFRAMES:
                    raise ValueError(
                        f"k_end={k_end} outside [1,{NFRAMES}] "
                        f"(pp={pp}, video={video}, t_end_s={te})"
                    )

                # --- logical consistency check ---
                if k_start > k_end:
                    raise ValueError(
                        f"k_start > k_end ({k_start} > {k_end}) "
                        f"(pp={pp}, video={video}, t_begin_s={tb}, t_end_s={te})"
                    )


                if k_end >= k_start:
                    frames[k_start-1:k_end] = dial  # last fixation wins

        df = pd.DataFrame({
            "pp": pp,
            "video": video,
            "frame": np.arange(1, NFRAMES + 1),
            "dial": frames
        })
        out_rows.append(df)

result = pd.concat(out_rows, ignore_index=True)

# Use nullable integer type: keeps missing as <NA> in pandas, blank in CSV
result["dial"] = result["dial"].astype("Int64")

# Optional ordering
result = result.sort_values(["pp", "video", "frame"]).reset_index(drop=True)


result.to_csv(
    "CatS_from_fixations_noDial7.csv",
    index=False,
    na_rep="nan"
)

print("Wrote CatS_from_fixations_noDial7.csv")
