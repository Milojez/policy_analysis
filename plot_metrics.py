"""
plot_metrics.py
===============
Plots training loss and validation accuracy from the CSV log.

Run:
    python plot_metrics.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config

df = pd.read_csv(config.LOG_PATH, on_bad_lines='skip')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f"Training curves — {config.MODEL_NAME}", fontsize=12)

# ── Train loss ────────────────────────────────────────────────────────────────
axes[0].plot(df['epoch'], df['train_loss'],
             color='steelblue', linewidth=1.5, marker='o', markersize=2,
             label='Total')
if 'train_dial_loss' in df.columns:
    axes[0].plot(df['epoch'], df['train_dial_loss'],
                 color='darkorange', linewidth=1.2, linestyle='--',
                 marker='o', markersize=2, label='Dial CE')
    axes[0].plot(df['epoch'], df['train_temp_loss'],
                 color='forestgreen', linewidth=1.2, linestyle='-.',
                 marker='o', markersize=2, label='Temporal MSE')
    axes[0].legend(fontsize=8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Train loss')
axes[0].grid(True, alpha=0.3)

# ── Validation accuracy ───────────────────────────────────────────────────────
axes[1].plot(df['epoch'], df['val_acc'] * 100,
             color='steelblue', linewidth=1.5, marker='o', markersize=2,
             label='Top-1 acc')
axes[1].plot(df['epoch'], df['val_top2_acc'] * 100,
             color='darkorange', linewidth=1.5, marker='o', markersize=2,
             label='Top-2 acc')
axes[1].axhline(100 / 6, color='red', linestyle='--', linewidth=1.2,
                label=f'Random baseline ({100/6:.1f}%)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Validation accuracy (video 6)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(config.CKPT_DIR, f"{config.MODEL_NAME}_metrics.png")
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

best_row = df.loc[df['val_acc'].idxmax()]
print(f"Best val_acc:  {best_row['val_acc']*100:.2f}%  at epoch {int(best_row['epoch'])}")
print(f"Best top2_acc: {df['val_top2_acc'].max()*100:.2f}%")
print(f"Last epoch:    {int(df['epoch'].iloc[-1])}  "
      f"loss={df['train_loss'].iloc[-1]:.4f}")
