"""
ablation.py
===========
Trains all 3 ablation variants sequentially and plots a comparison.

Variants:
  full        — fixation history + signal  (USE_FIXATIONS=True,  USE_SIGNAL=True)
  signal_only — signal only               (USE_FIXATIONS=False, USE_SIGNAL=True)
  fix_only    — fixation history only     (USE_FIXATIONS=True,  USE_SIGNAL=False)

Each variant gets its own checkpoint and log CSV under checkpoints/{name}/.
A combined plot is saved to results/ablation_comparison.png.

Run:
    python ablation.py
"""

import os
import sys
import csv
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import PolicyDataset
from model import PolicyNet

# ── Variant definitions ───────────────────────────────────────────────────────
VARIANTS = [
    {'name': 'full',        'use_fixations': True,  'use_signal': True},
    {'name': 'signal_only', 'use_fixations': False, 'use_signal': True},
    {'name': 'fix_only',    'use_fixations': True,  'use_signal': False},
]

RANDOM_BASELINE = 100.0 / 6   # ~16.7 %


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_splits():
    with open(config.DATA_PKL, 'rb') as f:
        payload = pickle.load(f)
    samples = payload['samples']
    train = [s for s in samples if s['video'] in config.TRAIN_VIDEOS]
    val   = [s for s in samples if s['video'] in config.VAL_VIDEOS]
    return train, val


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def top2_accuracy(logits, labels):
    top2 = logits.topk(2, dim=-1).indices
    return (top2 == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()


def evaluate(model, loader, device):
    model.eval()
    total_acc, total_top2, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pa  = batch['past_aois'].to(device)
            sig = batch['signal'].to(device)
            lbl = batch['label'].to(device)
            logits = model(pa, sig)
            bs = lbl.size(0)
            total_acc  += accuracy(logits, lbl)   * bs
            total_top2 += top2_accuracy(logits, lbl) * bs
            n += bs
    return total_acc / n, total_top2 / n


# ── Training one variant ──────────────────────────────────────────────────────

def train_variant(variant, train_ds, val_loader, device):
    name          = variant['name']
    use_fixations = variant['use_fixations']
    use_signal    = variant['use_signal']

    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", name)
    ckpt_path = os.path.join(ckpt_dir, f"{name}_best.pth")
    log_path  = os.path.join(ckpt_dir, f"{name}_log.csv")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Variant: {name}  "
          f"(fix={use_fixations}, sig={use_signal})")
    print(f"{'='*60}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)

    model = PolicyNet(
        use_fixations=use_fixations,
        use_signal=use_signal,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LR,
                                  weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    log_file   = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'val_acc', 'val_top2_acc'])

    best_val_acc = 0.0
    history = []   # (epoch, val_acc, val_top2)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            pa  = batch['past_aois'].to(device)
            sig = batch['signal'].to(device)
            lbl = batch['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(pa, sig), lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_acc, val_top2 = evaluate(model, val_loader, device)
        history.append((epoch + 1, val_acc * 100, val_top2 * 100))

        log_writer.writerow([epoch + 1, avg_loss, val_acc, val_top2])
        log_file.flush()

        print(f"  Ep {epoch+1:3d}/{config.EPOCHS}  "
              f"loss={avg_loss:.4f}  "
              f"val_acc={val_acc*100:.1f}%  "
              f"val_top2={val_top2*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': model.state_dict(),
                        'epoch': epoch,
                        'best_val_acc': best_val_acc}, ckpt_path)

    log_file.close()
    print(f"  Best val_acc: {best_val_acc*100:.2f}%  → {ckpt_path}")
    return history


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_comparison(all_histories):
    colors = {'full': 'steelblue', 'signal_only': 'darkorange', 'fix_only': 'forestgreen'}
    styles = {'full': '-', 'signal_only': '--', 'fix_only': '-.'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"PolicyNet ablation — {config.MODEL_NAME}\n"
        f"full: fix+sig | signal_only: sig only | fix_only: fix only",
        fontsize=11,
    )

    for name, history in all_histories.items():
        epochs = [h[0] for h in history]
        accs   = [h[1] for h in history]
        top2s  = [h[2] for h in history]
        col    = colors[name]
        sty    = styles[name]
        axes[0].plot(epochs, accs,  color=col, linestyle=sty, linewidth=1.8,
                     label=name)
        axes[1].plot(epochs, top2s, color=col, linestyle=sty, linewidth=1.8,
                     label=name)

    for ax, title in zip(axes, ['Val top-1 accuracy (%)', 'Val top-2 accuracy (%)']):
        ax.axhline(RANDOM_BASELINE, color='red', linestyle=':', linewidth=1.2,
                   label=f'Random ({RANDOM_BASELINE:.1f}%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_comparison.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_samples, val_samples = load_splits()
    print(f"Train: {len(train_samples):,}  Val: {len(val_samples):,}")

    train_ds   = PolicyDataset(train_samples)
    val_loader = DataLoader(PolicyDataset(val_samples),
                            batch_size=512, shuffle=False, num_workers=0)

    all_histories = {}
    for variant in VARIANTS:
        history = train_variant(variant, train_ds, val_loader, device)
        all_histories[variant['name']] = history

    plot_comparison(all_histories)

    # Summary table
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"{'Variant':<15}  {'Best val_acc':>12}  {'Best top2':>10}")
    for name, history in all_histories.items():
        best_acc  = max(h[1] for h in history)
        best_top2 = max(h[2] for h in history)
        print(f"{name:<15}  {best_acc:>11.2f}%  {best_top2:>9.2f}%")
    print(f"{'random':<15}  {RANDOM_BASELINE:>11.2f}%")


if __name__ == '__main__':
    main()
