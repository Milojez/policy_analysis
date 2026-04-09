"""
dataset.py
==========
PyTorch Dataset wrapper for pre-stacked .pt split files.

Loading from a .pt file is a single torch.load() call — much faster than
stacking 100k numpy arrays on every run.
"""

import os
import torch
from torch.utils.data import Dataset


class PolicyDataset(Dataset):
    """
    Each item:
      past_aois    : LongTensor  [N]       — AOI labels (0=pad, 1-6=dials)
      past_temporal: FloatTensor [N, 2]   — [duration_norm, saccade_norm] per past fixation
      signal       : FloatTensor [F, 6, 5] — signal features per (frame, dial)
      label        : LongTensor  []        — next AOI (0-indexed, 0..5)
      temporal     : FloatTensor [2]       — [saccade_norm, duration_norm] for next fixation

    Accepted input:
      - path to a pre-stacked .pt file
      - already loaded dict with keys:
            'past_aois', 'signals', 'labels', 'temporal'
    """

    def __init__(self, source):
        if isinstance(source, (str, bytes, os.PathLike)):
            data = torch.load(source, map_location="cpu", weights_only=True)
        else:
            data = source

        required = ["past_aois", "past_temporal", "signals", "future_signals", "labels", "temporal"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(
                f"PolicyDataset expected keys {required}, "
                f"but these are missing: {missing}. "
                f"Available keys: {list(data.keys())}"
            )

        self.past_aois      = data["past_aois"]       # [M, N]
        self.past_temporal  = data["past_temporal"]  # [M, N, 2]
        self.signals        = data["signals"]        # [M, F, 6, 5]
        self.future_signals = data["future_signals"] # [M, F_future, 6, 5]
        self.labels         = data["labels"]         # [M]
        self.temporal       = data["temporal"]       # [M, 2]

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "past_aois":      self.past_aois[idx],
            "past_temporal":  self.past_temporal[idx],
            "signal":         self.signals[idx],
            "future_signal":  self.future_signals[idx],
            "label":          self.labels[idx],
            "temporal":       self.temporal[idx],
        }