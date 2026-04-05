"""
evaluate.py
===========
Evaluate the best PolicyNet checkpoint on the test set.

Outputs (saved to config.RESULTS):
  - confusion_matrix.png   — normalised confusion matrix
  - metrics.txt            — accuracy, top-2 accuracy, per-dial recall

Run:
    python evaluate.py
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import PolicyDataset
from model import PolicyNet


def run_evaluation(model, loader, device, n_classes=6):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            pa  = batch['past_aois'].to(device)
            pt  = batch['past_temporal'].to(device)
            sig = batch['signal'].to(device)
            lbl = batch['label'].to(device)
            logits, _ = model(pa, sig, pt)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    recall = np.zeros(n_classes)
    for i in range(n_classes):
        if cm[i].sum() > 0:
            recall[i] = cm[i, i] / cm[i].sum()

    return acc, recall, cm, all_preds, all_labels


def plot_confusion_matrix(cm, out_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    dial_labels = [f"Dial {i+1}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(dial_labels, rotation=45, ha='right')
    ax.set_yticklabels(dial_labels)
    ax.set_xlabel("Predicted dial")
    ax.set_ylabel("True dial")
    ax.set_title("PolicyNet — Confusion matrix (normalised by row)\nTest video 7")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                    ha='center', va='center',
                    color='white' if cm_norm[i,j] > 0.5 else 'black',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(config.CKPT, map_location=device)
    model = PolicyNet(
        use_fixations=config.USE_FIXATIONS,
        use_signal=config.USE_SIGNAL,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"best_val_acc={ckpt.get('best_val_acc', float('nan'))*100:.2f}%")

    test_ds = PolicyDataset(config.TEST_PT)
    print(f"Test samples: {len(test_ds):,}")
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    acc, recall, cm, preds, labels = run_evaluation(model, test_loader, device)

    os.makedirs(config.RESULTS, exist_ok=True)

    print(f"\nTest accuracy:   {acc*100:.2f}%")
    print(f"Random baseline: {100/6:.2f}%")
    print("\nPer-dial recall:")
    for i, r in enumerate(recall):
        print(f"  Dial {i+1}: {r*100:.1f}%")

    metrics_path = os.path.join(config.RESULTS, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Test accuracy:   {acc*100:.2f}%\n")
        f.write(f"Random baseline: {100/6:.2f}%\n\n")
        f.write("Per-dial recall:\n")
        for i, r in enumerate(recall):
            f.write(f"  Dial {i+1}: {r*100:.2f}%\n")
        f.write("\nConfusion matrix (counts):\n")
        f.write(str(cm) + "\n")
    print(f"Saved: {metrics_path}")

    cm_path = os.path.join(config.RESULTS, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)


if __name__ == '__main__':
    main()
