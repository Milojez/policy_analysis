import numpy as np

def signal_for_dial(t_end: float, rg: int, use_matlab_std: bool = True) -> np.ndarray:
    """
    Port of MATLAB signal_for_dial.m
    Returns alldata (L,6) in radians, bandwidth order low->high.
    """
    rs = np.random.RandomState(rg)  # MT19937 like MATLAB rand

    Fs = 50
    L = int(t_end * Fs)
    t = np.arange(L) / Fs

    cutoff = np.array([.03, .05, .12, .20, .32, .48], dtype=float)
    c = np.unique(np.concatenate([np.logspace(np.log10(0.001), np.log10(1), 40), cutoff]))

    alldata = np.zeros((L, 6), dtype=float)
    ddof = 1 if use_matlab_std else 0

    for j in range(6):
        cc = c[c <= cutoff[j]]
        d = 9999 * rs.rand(len(cc))
        y = np.sin(2*np.pi*cc[:, None] * (t[None, :] + d[:, None]))
        yn = y.sum(axis=0)
        yn = (yn - yn.mean()) / yn.std(ddof=ddof)
        alldata[:, j] = yn

    return np.pi * alldata / 3.59247397659283


def build_signals_all_videos(t_end: float = 90.0, use_matlab_std: bool = True) -> np.ndarray:
    """Returns signals with shape (7,4500,6)."""
    return np.stack([signal_for_dial(t_end, rg, use_matlab_std) for rg in range(1, 8)], axis=0)
