import numpy as np

def compute_fig9_curves(signals: np.ndarray, CatHb: np.ndarray, fps: int = 50):
    """
    signals: (7,4500,6) radians, bw low->high
    CatHb:   (4500,7,6) percent participants
    Returns centers + curves for Fig 9: NSI, NVSI, NTSI
    """
    temp2 = np.vstack(signals)                 # (31500,6)
    temp3 = np.vstack([CatHb[:, v, :] for v in range(7)])  # (31500,6)

    temp4 = np.vstack([np.full((1,6), np.nan), np.diff(temp2, axis=0) * fps])  # rad/s
    temp5 = temp2 / (-temp4)  # TTC

    IV  = np.arange(-np.pi, np.pi + np.deg2rad(5), np.deg2rad(5))  # rad bins
    IV3 = np.arange(-100, 100.5, 0.5)                              # s bins

    NSI  = np.full((len(IV)-1, 6), np.nan)
    NVSI = np.full((len(IV)-1, 6), np.nan)
    NTSI = np.full((len(IV3)-1, 6), np.nan)

    for bw in range(6):
        for i in range(len(IV)-1):
            lo, hi = IV[i], IV[i+1]
            idx_a = np.where((temp2[:, bw] > lo) & (temp2[:, bw] <= hi))[0]
            idx_v = np.where((temp4[:, bw] > lo) & (temp4[:, bw] <= hi))[0]
            if idx_a.size > 250:
                NSI[i, bw] = np.mean(temp3[idx_a, bw])
            if idx_v.size > 250:
                NVSI[i, bw] = np.mean(temp3[idx_v, bw])

        for i in range(len(IV3)-1):
            lo, hi = IV3[i], IV3[i+1]
            idx_t = np.where((temp5[:, bw] > lo) & (temp5[:, bw] <= hi))[0]
            if idx_t.size > 250:
                NTSI[i, bw] = np.mean(temp3[idx_t, bw])

    angle_c = np.rad2deg(IV[:-1] + np.diff(IV)/2)
    vel_c   = angle_c.copy()
    ttc_c   = IV3[:-1] + np.diff(IV3)/2

    return angle_c, vel_c, ttc_c, NSI, NVSI, NTSI
