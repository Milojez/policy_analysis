"""
build_dataset.py
================
Builds the PolicyNet dataset from:
  - GT_all_fixation_noDial7_noPPs.csv  (ground-truth fixations)
  - movie_0{N}_dial_signals.csv         (per-dial angle/speed signals)

Samples are built with stride N (non-overlapping windows):
  - For K fixations per (pp, video), targets are at indices N, 2N, 3N, ...
  - past_aois = the N fixations immediately before the target (no padding)
  - Consecutive samples share zero fixations in their past_aois context.
  - signal window      [F, 6, 4]  (F evenly-spaced frames ending at t_next)
  - next AOI label     (0-indexed, 0..5)
  - saccade_norm       normalised log1p saccade time (s)
  - duration_norm      normalised log1p fixation duration (s)

needle_to_threshold_norm:
  Equivalent to the Euclidean approach in extend_with_dial_signals.py,
  simplified for unit-radius needle/threshold on the same circle:
      norms[t] = sqrt(max(2 - 2*cos(angle[t]), 0))

Urgency formula (same as heatmap_ar_model/dataset.py):
      raw_rate  = max(prev_norm - norms[t], 0)
      urgency   = tanh(raw_rate / (norms[t] + 0.05) / urgency_std)

Run:
    python build_dataset.py
"""

import os
import sys
import math
import pickle
import re
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import config

URGENCY_EPS = 0.05


# ── Signal CSV helpers ─────────────────────────────────────────────────────────

def parse_dial_id_to_pos(columns):
    mapping = {}
    for col in columns:
        m = re.match(r'angle_dial_id_(\d+)_pos_(\d+)', col)
        if m:
            mapping[int(m.group(1))] = int(m.group(2))
    return mapping


def load_signal_csv(video_num):
    path = os.path.join(config.SIG_DIR, f"movie_0{video_num}_dial_signals.csv")
    df = pd.read_csv(path)
    id_to_pos = parse_dial_id_to_pos(df.columns)
    angle_cols = {}
    for col in df.columns:
        m = re.match(r'angle_dial_id_(\d+)_pos_(\d+)', col)
        if m:
            pos = int(m.group(2))
            angle_cols[pos] = col
    positions = sorted(angle_cols.keys())
    return df, id_to_pos, positions, angle_cols


def floor_lookup_index(time_end_array, query_time):
    idx = np.searchsorted(time_end_array, query_time, side='right') - 1
    return max(int(idx), 0)


def build_signal_window(df, time_end_arr, angle_cols, t_next, urgency_std):
    """
    Build a [F, 6, 4] signal window ending at t_next.
    Features per (frame, dial): [sin(angle), cos(angle), urgency, distance]
    """
    F = round(config.SIGNAL_LENGTH_S * config.SIGNAL_HZ)
    t_start = t_next - config.SIGNAL_LENGTH_S
    query_times = np.linspace(t_start, t_next, F)

    row_indices = [floor_lookup_index(time_end_arr, qt) for qt in query_times]

    positions = sorted(angle_cols.keys())
    window = np.zeros((F, 6, 4), dtype=np.float32)

    for dial_slot, pos in enumerate(positions):
        col = angle_cols[pos]
        angles = df[col].values.astype(np.float64)

        for fi, row_idx in enumerate(row_indices):
            angle = angles[row_idx]
            norm_val = math.sqrt(max(2.0 - 2.0 * math.cos(angle), 0.0))

            prev_idx = max(row_idx - 1, 0)
            prev_norm = math.sqrt(max(2.0 - 2.0 * math.cos(angles[prev_idx]), 0.0))
            raw_rate = max(prev_norm - norm_val, 0.0)
            urgency  = math.tanh(raw_rate / (norm_val + URGENCY_EPS) / urgency_std)

            window[fi, dial_slot, 0] = math.sin(angle)
            window[fi, dial_slot, 1] = math.cos(angle)
            window[fi, dial_slot, 2] = urgency
            window[fi, dial_slot, 3] = 1.0 - norm_val

    return window


# ── AOI verification ───────────────────────────────────────────────────────────

def check_aoi_match(x, y, dial_pos):
    if dial_pos not in config.AOI_BY_POSITION:
        return False
    x0, x1, y0, y1 = config.AOI_BY_POSITION[dial_pos]
    return (x > x0) and (x <= x1) and (y >= y0) and (y < y1)


def verify_aoi_assignments(gt_df):
    print("\nAOI coordinate verification:")
    for pos in sorted(config.AOI_BY_POSITION.keys()):
        sub = gt_df[gt_df['dial'] == pos]
        if len(sub) == 0:
            continue
        matches = sub.apply(
            lambda r: check_aoi_match(r['x_fix'], r['y_fix'], pos), axis=1
        )
        mismatch_rate = (~matches).mean() * 100
        print(f"  Dial {pos}: {len(sub):6d} fixations, "
              f"mismatch={mismatch_rate:.1f}%")
    print()


# ── urgency_std from training data ────────────────────────────────────────────

def compute_urgency_std(train_videos):
    all_raw_rates = []
    for vnum in train_videos:
        df, _, positions, angle_cols = load_signal_csv(vnum)
        for pos in positions:
            col = angle_cols[pos]
            angles = df[col].values.astype(np.float64)
            norms = np.sqrt(np.maximum(2.0 - 2.0 * np.cos(angles), 0.0))
            prev_norms = np.concatenate([[norms[0]], norms[:-1]])
            raw_rates = np.maximum(prev_norms - norms, 0.0)
            urgency_raw = raw_rates / (norms + URGENCY_EPS)
            all_raw_rates.extend(urgency_raw.tolist())
    std = float(np.std(all_raw_rates))
    return max(std, 1e-6)


# ── Temporal normalisation stats ──────────────────────────────────────────────

def fit_temporal_stats(train_samples):
    """
    Fit log1p normalisation stats for saccade and duration
    using training samples only.
    """
    saccades  = np.log1p(np.array([s['saccade_s']  for s in train_samples],
                                   dtype=np.float64))
    durations = np.log1p(np.array([s['duration_s'] for s in train_samples],
                                   dtype=np.float64))
    return {
        'saccade_log_mean':  float(saccades.mean()),
        'saccade_log_std':   float(max(saccades.std(), 1e-6)),
        'duration_log_mean': float(durations.mean()),
        'duration_log_std':  float(max(durations.std(), 1e-6)),
    }


def norm_saccade(val_s, stats):
    return (math.log1p(max(val_s, 0.0)) - stats['saccade_log_mean']) / stats['saccade_log_std']

def norm_duration(val_s, stats):
    return (math.log1p(max(val_s, 0.0)) - stats['duration_log_mean']) / stats['duration_log_std']

def denorm_saccade(val_norm, stats):
    return max(math.expm1(val_norm * stats['saccade_log_std'] + stats['saccade_log_mean']), 0.0)

def denorm_duration(val_norm, stats):
    return max(math.expm1(val_norm * stats['duration_log_std'] + stats['duration_log_mean']), 0.001)


# ── Main dataset builder ───────────────────────────────────────────────────────

def build_samples_raw(gt_df, signal_data, urgency_std):
    """
    Build samples without temporal normalisation (raw saccade_s / duration_s).
    Temporal normalisation is applied after fitting stats on training split.
    """
    N = config.PAST_FIXATIONS
    samples = []

    for (pp, video), grp in gt_df.groupby(['pp', 'video']):
        grp = grp.sort_values('t_begin_s').reset_index(drop=True)
        dials      = grp['dial'].tolist()
        t_begins   = grp['t_begin_s'].tolist()
        t_ends     = grp['t_end_s'].tolist()
        durations  = grp['duration_s'].tolist()

        if video not in signal_data:
            continue
        df_sig, time_end_arr, angle_cols = signal_data[video]

        # Precompute saccade for every fixation in this trial
        saccade_list = [0.0] + [
            max(t_begins[j] - t_ends[j - 1], 0.0) for j in range(1, len(dials))
        ]

        # Stride by config.STRIDE. Starting at N ensures every sample has a
        # full history window (no padding needed).
        for i in range(N, len(dials), config.STRIDE):
            next_aoi   = dials[i]
            t_next     = t_begins[i]
            saccade_s  = saccade_list[i]
            duration_s = durations[i]

            past_aois      = dials[i - N: i]        # exactly N, no padding needed
            past_durations = durations[i - N: i]
            past_saccades  = saccade_list[i - N: i]

            window = build_signal_window(df_sig, time_end_arr, angle_cols,
                                         t_next, urgency_std)

            samples.append({
                'past_aois':      past_aois,
                'past_durations': past_durations,
                'past_saccades':  past_saccades,
                'signal':         window,
                'label':          next_aoi - 1,      # 0-indexed
                'saccade_s':      saccade_s,
                'duration_s':     duration_s,
                'pp':             pp,
                'video':          video,
                't_next':         t_next,
                't_end':          t_ends[i],
            })

    return samples


def main():
    print("Loading GT fixation CSV...")
    gt_df = pd.read_csv(config.GT_CSV)
    print(f"  {len(gt_df):,} fixations | "
          f"pp: {gt_df['pp'].nunique()} | "
          f"videos: {sorted(gt_df['video'].unique())}")

    verify_aoi_assignments(gt_df)

    print("Computing urgency_std from training videos...")
    urgency_std = compute_urgency_std(config.TRAIN_VIDEOS)
    print(f"  urgency_std = {urgency_std:.6f}")

    print("Loading signal CSVs...")
    all_videos = sorted(gt_df['video'].unique())
    signal_data = {}
    for vnum in all_videos:
        df, _, positions, angle_cols = load_signal_csv(vnum)
        time_end_arr = df['time_end'].values.astype(np.float64)
        signal_data[vnum] = (df, time_end_arr, angle_cols)
        print(f"  Video {vnum}: {len(df)} signal frames, "
              f"dials at positions {positions}")

    print("\nBuilding raw samples...")
    samples = build_samples_raw(gt_df, signal_data, urgency_std)
    print(f"  Total raw samples: {len(samples):,}")

    # Split
    train = [s for s in samples if s['video'] in config.TRAIN_VIDEOS]
    val   = [s for s in samples if s['video'] in config.VAL_VIDEOS]
    test  = [s for s in samples if s['video'] in config.TEST_VIDEOS]
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    # Fit temporal stats on training data only
    print("Fitting temporal normalisation stats from training split...")
    temporal_stats = fit_temporal_stats(train)
    print(f"  saccade  log_mean={temporal_stats['saccade_log_mean']:.4f}  "
          f"log_std={temporal_stats['saccade_log_std']:.4f}")
    print(f"  duration log_mean={temporal_stats['duration_log_mean']:.4f}  "
          f"log_std={temporal_stats['duration_log_std']:.4f}")

    # Apply normalisation (target + past fixation temporals)
    for s in samples:
        s['saccade_norm']  = norm_saccade(s['saccade_s'],  temporal_stats)
        s['duration_norm'] = norm_duration(s['duration_s'], temporal_stats)
        s['past_temporal'] = [
            [norm_duration(d, temporal_stats), norm_saccade(sc, temporal_stats)]
            for d, sc in zip(s['past_durations'], s['past_saccades'])
        ]

    os.makedirs(os.path.dirname(config.DATA_PKL), exist_ok=True)

    # Save full pkl (used by visualize.py for autoregressive generation)
    payload = {
        'samples':        samples,
        'urgency_std':    urgency_std,
        'temporal_stats': temporal_stats,
        'F':              round(config.SIGNAL_LENGTH_S * config.SIGNAL_HZ),
        'N':              config.PAST_FIXATIONS,
    }
    with open(config.DATA_PKL, 'wb') as f:
        pickle.dump(payload, f)
    print(f"Saved: {config.DATA_PKL}")

    # Save small metadata pkl (stats only — loaded by train/evaluate)
    import torch
    meta = {'urgency_std': urgency_std, 'temporal_stats': temporal_stats}
    with open(config.META_PKL, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved: {config.META_PKL}")

    # Save pre-stacked split tensors for fast training
    print("Saving pre-stacked split tensors...")
    for split_samples, path, name in [
        (train, config.TRAIN_PT, 'train'),
        (val,   config.VAL_PT,   'val'),
        (test,  config.TEST_PT,  'test'),
    ]:
        past_arr  = np.array([s['past_aois']     for s in split_samples], dtype=np.int64)
        pt_arr    = np.array([s['past_temporal'] for s in split_samples], dtype=np.float32)
        sig_arr   = np.stack([s['signal']        for s in split_samples], axis=0)
        lbl_arr   = np.array([s['label']         for s in split_samples], dtype=np.int64)
        temp_arr  = np.array([[s['saccade_norm'], s['duration_norm']]
                               for s in split_samples], dtype=np.float32)
        torch.save({
            'past_aois':     torch.from_numpy(past_arr),
            'past_temporal': torch.from_numpy(pt_arr),
            'signals':       torch.from_numpy(sig_arr),
            'labels':        torch.from_numpy(lbl_arr),
            'temporal':      torch.from_numpy(temp_arr),
        }, path)
        print(f"  Saved {name}: {len(split_samples):,} samples → {path}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
