"""
EDGE-DR — Training Pipeline
=============================
Full training loop with:
  - Focal Loss + class-balanced sampling
  - Cosine annealing LR scheduler with warmup
  - Mixed precision (AMP)
  - Best-checkpoint saving (QWK)
  - TensorBoard logging

Usage:
    python -m src.models.train --config configs/config.yaml
"""

import os
import time
import yaml
import click
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None
from pathlib import Path
from tqdm import tqdm

from src.data.dataset import get_dataloaders
from src.models.model import (
    EdgeDRModel, FocalLoss,
    quadratic_weighted_kappa, compute_sensitivity_specificity
)


# ──────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────

def build_loss(cfg: dict, class_weights: torch.Tensor) -> nn.Module:
    if cfg["training"]["loss"] == "focal":
        alpha = class_weights if cfg["training"]["focal_alpha"] == "auto" else None
        return FocalLoss(gamma=cfg["training"]["focal_gamma"], alpha=alpha)
    else:
        return nn.CrossEntropyLoss(weight=class_weights,
                                   label_smoothing=cfg["model"]["label_smoothing"])


def run_epoch(model, loader, criterion, optimizer, scaler,
              device, is_train: bool):
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in tqdm(loader, leave=False,
                                  desc="Train" if is_train else "Val"):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=scaler is not None):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if is_train and optimizer:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    preds_arr   = np.array(all_preds)
    targets_arr = np.array(all_targets)
    qwk = quadratic_weighted_kappa(preds_arr, targets_arr)
    sens_spec = compute_sensitivity_specificity(preds_arr, targets_arr)
    accuracy = (preds_arr == targets_arr).mean()

    return {
        "loss":        avg_loss,
        "qwk":         qwk,
        "accuracy":    accuracy,
        "sensitivity": sens_spec["sensitivity"],
        "specificity": sens_spec["specificity"],
    }


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────
    csv_path = Path(cfg["data"]["processed_dir"]) / "dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}. "
            "Run: python -m src.data.preprocess first."
        )

    loaders = get_dataloaders(
        csv_path=str(csv_path),
        image_size=cfg["data"]["model_input_size"],
        batch_size=cfg["training"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        num_workers=0 if os.name == "nt" else min(4, os.cpu_count() or 1),
        seed=cfg["data"]["random_seed"],
    )
    print(f"Train: {loaders['num_train']:,} | Val: {loaders['num_val']:,} | "
          f"Test: {loaders['num_test']:,}")

    # ── Model ─────────────────────────────────
    model = EdgeDRModel(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {cfg['model']['backbone']} - {total_params:.2f}M parameters")

    # ── Loss ──────────────────────────────────
    class_weights = loaders["class_weights"].to(device)
    criterion = build_loss(cfg, class_weights)

    # ── Optimizer ─────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ── Scheduler: linear warmup + cosine decay ──
    warmup_sched = LinearLR(optimizer,
                             start_factor=0.1, total_iters=cfg["training"]["warmup_epochs"])
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"] - cfg["training"]["warmup_epochs"],
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[cfg["training"]["warmup_epochs"]],
    )

    # ── AMP ───────────────────────────────────
    scaler = GradScaler() if cfg["training"]["mixed_precision"] and device.type == "cuda" else None

    # ── Logging ───────────────────────────────
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb_logs")) if HAS_TENSORBOARD else None

    best_qwk   = -1.0
    best_epoch = 0

    # ── Training loop ─────────────────────────
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, loaders["train"], criterion,
                                   optimizer, scaler, device, is_train=True)
        val_metrics   = run_epoch(model, loaders["val"],   criterion,
                                   None,      None,  device, is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # TensorBoard
        if writer:
            for split, m in [("train", train_metrics), ("val", val_metrics)]:
                for k, v in m.items():
                    writer.add_scalar(f"{split}/{k}", v, epoch)
            writer.add_scalar("lr", lr, epoch)

        print(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']} "
            f"[{elapsed:.0f}s] lr={lr:.2e} | "
            f"Train QWK={train_metrics['qwk']:.4f} loss={train_metrics['loss']:.4f} | "
            f"Val QWK={val_metrics['qwk']:.4f} Sens={val_metrics['sensitivity']:.3f} "
            f"Spec={val_metrics['specificity']:.3f}"
        )

        # Save best checkpoint
        if val_metrics["qwk"] > best_qwk:
            best_qwk   = val_metrics["qwk"]
            best_epoch = epoch
            ckpt_path  = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "val_qwk":      best_qwk,
                "config":       cfg,
            }, ckpt_path)
            print(f"  * New best checkpoint saved (QWK={best_qwk:.4f})")

    print(f"\nTraining complete. Best epoch: {best_epoch} | QWK: {best_qwk:.4f}")
    if writer:
        writer.close()

    # ── Final test evaluation ──────────────────
    best_ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = run_epoch(model, loaders["test"], criterion,
                              None, None, device, is_train=False)
    print(
        f"\nTest Set Results:\n"
        f"  QWK:         {test_metrics['qwk']:.4f}  (target >0.85)\n"
        f"  Sensitivity: {test_metrics['sensitivity']:.4f}  (target >0.92)\n"
        f"  Specificity: {test_metrics['specificity']:.4f}  (target >0.88)\n"
        f"  Accuracy:    {test_metrics['accuracy']:.4f}"
    )

    return model


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

@click.command()
@click.option("--config", default="configs/config.yaml")
def main(config):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()

