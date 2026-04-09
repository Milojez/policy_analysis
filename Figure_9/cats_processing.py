import numpy as np
import pandas as pd

DIAL_CONFIG = np.array([
    [3,6,2,4,5,1],
    [4,6,1,5,2,3],
    [1,3,4,5,6,2],
    [5,3,2,6,1,4],
    [6,2,1,3,4,5],
    [3,5,2,4,1,6],
    [5,1,4,3,2,6],
], dtype=int)

def load_cats_csv(path: str, NrPP: int = 92, n_videos: int = 7, n_frames: int = 4500):
    df = pd.read_csv(path)

    CatS = np.full((NrPP, n_videos, n_frames), np.nan, dtype=float)

    df["dial"] = pd.to_numeric(df["dial"], errors="coerce")

    CatS[
        df["pp"].to_numpy() - 1,
        df["video"].to_numpy() - 1,
        df["frame"].to_numpy() - 1
    ] = df["dial"].to_numpy()

    return CatS

def dialpos_to_bw_rank(CatS: np.ndarray, dial_config: np.ndarray = DIAL_CONFIG) -> np.ndarray:
    """
    Convert CatS values that are dial positions (1..6 in screen order)
    to bandwidth rank (1..6 low->high), per video.
    """
    NrPP, n_videos, _ = CatS.shape
    CatSb = np.full_like(CatS, np.nan)

    for v in range(n_videos):
        # b[bw-1] = dial_pos (1..6)
        b = np.argsort(dial_config[v, :]) + 1
        inv = np.zeros(7, dtype=int)  # inv[dial_pos] = bw
        for bw in range(1, 7):
            inv[b[bw-1]] = bw

        x = CatS[:, v, :]
        y = np.full_like(x, np.nan)
        for dial_pos in range(1, 7):
            y[x == dial_pos] = inv[dial_pos]
        CatSb[:, v, :] = y

    return CatSb

def cats_to_CatHb(CatSb: np.ndarray) -> np.ndarray:
    """
    CatHb[frame, video, bw] = % participants looking at bw (bw=1..6 -> index 0..5).
    """
    NrPP, n_videos, n_frames = CatSb.shape
    CatHb = np.full((n_frames, n_videos, 6), np.nan, dtype=float)

    for v in range(n_videos):
        X = CatSb[:, v, :]  # (pp, frame)
        valid_pp = np.any(~np.isnan(X), axis=1)
        denom = valid_pp.sum()
        if denom == 0:
            continue
        vals = X[valid_pp, :]  # (valid_pp, frame)
        for bw in range(1, 7):
            CatHb[:, v, bw-1] = 100.0 * np.nanmean(vals == bw, axis=0)

    return CatHb

