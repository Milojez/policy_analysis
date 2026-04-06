"""
logistic_baseline.py
====================
Logistic regression baseline for next-dial prediction.

Uses per-dial summary features extracted from the signal window:
  - current state  (last frame): urgency, distance, speed_norm
  - recent average (all frames): urgency, speed_norm
  - peak urgency over window
  - speed trend: speed_norm[-1] - speed_norm[0]

6 features × 6 dials = 36 signal features.
Optionally appends one-hot encoded past_aois.

This baseline answers: how much of the prediction can a linear model
find from the signal alone? If the neural network barely beats this,
the task ceiling is low. If it beats it clearly, the network is
learning something beyond simple signal thresholding.

Run:
    python logistic_baseline.py
"""

import os
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
import config


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(signals, past_aois, include_fixations=True):
    """
    signals:   np.ndarray [M, F, 6, 5]  — signal windows
    past_aois: np.ndarray [M, N]        — past AOI labels (0=pad, 1-6)

    Returns X: np.ndarray [M, n_features]

    Signal features per dial (36 total):
        urgency_last, distance_last, speed_last,   ← current state
        urgency_mean, speed_mean,                  ← recent average
        urgency_max,                               ← peak urgency
        speed_trend                                ← speed_norm[-1] - speed_norm[0]
    """
    M, F, D, C = signals.shape

    # Feature indices in signal: [sin, cos, urgency, distance, speed_norm]
    IDX_URG  = 2
    IDX_DIST = 3
    IDX_SPD  = 4

    feats = []

    for d in range(D):
        urg  = signals[:, :, d, IDX_URG]    # [M, F]
        dist = signals[:, :, d, IDX_DIST]   # [M, F]
        spd  = signals[:, :, d, IDX_SPD]    # [M, F]

        feats.append(urg[:, -1])             # urgency at last frame
        feats.append(dist[:, -1])            # distance at last frame
        feats.append(spd[:, -1])             # speed at last frame
        feats.append(urg.mean(axis=1))       # mean urgency over window
        feats.append(spd.mean(axis=1))       # mean speed over window
        feats.append(urg.max(axis=1))        # peak urgency
        feats.append(spd[:, -1] - spd[:, 0])  # speed trend

    X_sig = np.stack(feats, axis=1)          # [M, 7*6=42]

    if not include_fixations:
        return X_sig

    # One-hot encode past_aois (0=pad, 1-6 dials → 7 categories per step)
    N = past_aois.shape[1]
    one_hots = []
    for n in range(N):
        oh = np.zeros((M, 7), dtype=np.float32)
        for m in range(M):
            oh[m, int(past_aois[m, n])] = 1.0
        one_hots.append(oh)
    X_fix = np.concatenate(one_hots, axis=1)  # [M, 7*N]

    return np.concatenate([X_sig, X_fix], axis=1)


def load_split(pt_path):
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    signals   = data["signals"].numpy()     # [M, F, 6, 5]
    past_aois = data["past_aois"].numpy()   # [M, N]
    labels    = data["labels"].numpy()      # [M]
    return signals, past_aois, labels


def top2_accuracy(proba, labels):
    top2 = np.argsort(proba, axis=1)[:, -2:]
    return np.mean([labels[i] in top2[i] for i in range(len(labels))])


def per_dial_recall(preds, labels, n_classes=6):
    recall = np.zeros(n_classes)
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            recall[i] = (preds[mask] == i).mean()
    return recall


# ── Main ──────────────────────────────────────────────────────────────────────

def run(include_fixations):
    print(f"\n{'='*55}")
    label = "signal + fixation history" if include_fixations else "signal only"
    print(f"Logistic regression — {label}")
    print('='*55)

    print("Loading splits...")
    sig_tr, pa_tr, y_tr = load_split(config.TRAIN_PT)
    sig_va, pa_va, y_va = load_split(config.VAL_PT)
    sig_te, pa_te, y_te = load_split(config.TEST_PT)

    print(f"  Train: {len(y_tr):,}  Val: {len(y_va):,}  Test: {len(y_te):,}")

    print("Extracting features...")
    X_tr = extract_features(sig_tr, pa_tr, include_fixations)
    X_va = extract_features(sig_va, pa_va, include_fixations)
    X_te = extract_features(sig_te, pa_te, include_fixations)
    print(f"  Feature dim: {X_tr.shape[1]}")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    print("Training logistic regression...")
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    for split_name, X, y in [("Val", X_va, y_va), ("Test", X_te, y_te)]:
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        acc   = accuracy_score(y, preds)
        top2  = top2_accuracy(proba, y)
        recall = per_dial_recall(preds, y)

        print(f"\n{split_name}:")
        print(f"  Accuracy:        {acc*100:.2f}%")
        print(f"  Top-2 accuracy:  {top2*100:.2f}%")
        print(f"  Random baseline: {100/6:.2f}%")
        print(f"  Per-dial recall:")
        for i, r in enumerate(recall):
            print(f"    Dial {i+1}: {r*100:.1f}%")


def main():
    print("Random baseline: {:.2f}%".format(100 / 6))

    # Signal only — pure signal reliance, no fixation history
    run(include_fixations=False)

    # Signal + fixation history — gives the logistic model everything
    run(include_fixations=True)


if __name__ == "__main__":
    main()
