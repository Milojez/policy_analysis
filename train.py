"""
train.py
========
Training loop for PolicyNet.

Loss:
    L = CrossEntropy(dial_logits, next_aoi)
      + LAMBDA_TEMPORAL * MSE(temporal_pred, [saccade_norm, duration_norm])

Best checkpoint is saved immediately whenever validation CE improves,
overwriting the previous best checkpoint.

Run:
    python train.py
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import config
from dataset import PolicyDataset
from model import PolicyNet


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def top2_accuracy(logits, labels):
    top2 = logits.topk(2, dim=-1).indices
    return (top2 == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()


def evaluate(model, loader, device, ce_loss):
    model.eval()
    total_acc, total_top2, total_ce, n = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for batch in loader:
            pa  = batch['past_aois'].to(device)
            pt  = batch['past_temporal'].to(device)
            sig = batch['signal'].to(device)
            lbl = batch['label'].to(device)

            logits, _ = model(pa, sig, pt)

            bs = lbl.size(0)
            total_acc  += accuracy(logits, lbl) * bs
            total_top2 += top2_accuracy(logits, lbl) * bs
            total_ce   += ce_loss(logits, lbl).item() * bs
            n += bs

    return total_ce / n, total_acc / n, total_top2 / n


def main():
    torch.manual_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = PolicyDataset(config.TRAIN_PT)
    val_ds   = PolicyDataset(config.VAL_PT)
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    n_train = len(train_ds)
    n_batches_ep = len(train_loader)
    print(f"Batches/epoch: {n_batches_ep}  "
          f"(batch_size={config.BATCH_SIZE}, "
          f"total_train_samples={n_train:,})")

    model = PolicyNet(
        use_fixations=config.USE_FIXATIONS,
        use_signal=config.USE_SIGNAL,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    ce_loss = nn.CrossEntropyLoss()

    # Cosine annealing after linear warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.EPOCHS - config.WARMUP_EPOCHS, 1),
        eta_min=config.LR_MIN,
    )

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS, exist_ok=True)

    log_exists = os.path.exists(config.LOG_PATH)
    log_file = open(config.LOG_PATH, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            'epoch',
            'train_loss',
            'train_dial_loss',
            'train_temp_loss',
            'val_ce',
            'val_acc',
            'val_top2_acc',
            'lr',
        ])

    best_val_ce = float('inf')
    best_val_acc = 0.0

    for epoch in range(config.EPOCHS):
        # ── LR schedule: linear warmup then cosine annealing ────────────────
        if epoch < config.WARMUP_EPOCHS:
            lr_scale = (epoch + 1) / max(config.WARMUP_EPOCHS, 1)
            for pg in optimizer.param_groups:
                pg['lr'] = config.LR * lr_scale
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        model.train()
        total_loss, total_dial, total_temp, n_batches = 0.0, 0.0, 0.0, 0

        for step, batch in enumerate(train_loader):
            pa      = batch['past_aois'].to(device)
            pt      = batch['past_temporal'].to(device)
            sig     = batch['signal'].to(device)
            lbl     = batch['label'].to(device)
            temp_gt = batch['temporal'].to(device)

            optimizer.zero_grad()
            logits, temp_pred = model(pa, sig, pt)

            loss_dial = ce_loss(logits, lbl)
            loss_temp = F.mse_loss(temp_pred, temp_gt)
            loss = loss_dial + config.LAMBDA_TEMPORAL * loss_temp

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dial += loss_dial.item()
            total_temp += loss_temp.item()
            n_batches += 1

            samples_done = min((step + 1) * config.BATCH_SIZE, n_train)
            print(f"\r  Epoch {epoch+1:3d}/{config.EPOCHS}  "
                  f"[{samples_done:>{len(str(n_train))}}/{n_train:,}  "
                  f"batch {step+1}/{n_batches_ep}  "
                  f"loss={loss.item():.4f}]",
                  end='', flush=True)

        avg_loss = total_loss / max(n_batches, 1)
        avg_dial = total_dial / max(n_batches, 1)
        avg_temp = total_temp / max(n_batches, 1)

        val_ce, val_acc, val_top2 = evaluate(model, val_loader, device, ce_loss)

        print(f"\rEpoch {epoch+1:3d}/{config.EPOCHS}  "
              f"loss={avg_loss:.4f} (dial={avg_dial:.4f} temp={avg_temp:.4f})  "
              f"val_ce={val_ce:.4f}  "
              f"val_acc={val_acc*100:.1f}%  val_top2={val_top2*100:.1f}%  "
              f"lr={current_lr:.2e}")

        log_writer.writerow([
            epoch + 1,
            avg_loss,
            avg_dial,
            avg_temp,
            val_ce,
            val_acc,
            val_top2,
            current_lr,
        ])
        log_file.flush()

        # Save best model immediately, overwriting previous best checkpoint
        if val_ce < best_val_ce:
            best_val_ce = val_ce
            best_val_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'best_val_ce': best_val_ce,
                'best_val_acc': best_val_acc,
            }, config.CKPT)
            print(f"  ✓ Best model saved "
                  f"(val_ce={best_val_ce:.4f}, val_acc={best_val_acc*100:.2f}%)")

    log_file.close()
    print(f"\nTraining complete. Best val_ce: {best_val_ce:.4f}")
    print(f"Checkpoint: {config.CKPT}")
    print(f"Log:        {config.LOG_PATH}")


if __name__ == '__main__':
    main()