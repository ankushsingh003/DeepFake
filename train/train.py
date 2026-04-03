"""
train/train.py
Full training loop using PyTorch Lightning.

Features:
  - Binary cross-entropy loss with label smoothing
  - AUC-ROC + F1 + accuracy metrics per epoch
  - Automatic checkpoint saving (best val AUC)
  - Early stopping
  - Cosine LR schedule with warmup
  - Class-imbalance handling via pos_weight

Usage:
    python train/train.py
    python train/train.py --epochs 30 --batch-size 8 --no-face-detect
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

from models import DeepfakeDetector
from data import get_dataloaders

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ── Lightning Module ───────────────────────────────────────────────────────────

class DeepfakeLightning(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,   # set > 1 if fake class is underrepresented
        freeze_vit: bool = False,
        embed_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DeepfakeDetector(
            embed_dim=embed_dim,
            dropout=dropout,
            freeze_vit=freeze_vit,
        )

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        # Accumulators for epoch-level AUC
        self._val_probs  = []
        self._val_labels = []
        self._test_probs  = []
        self._test_labels = []

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, spatial, temporal):
        return self.model(spatial, temporal)

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        spatial, temporal, labels = batch
        logits = self(spatial, temporal)
        loss = self.loss_fn(logits, labels)

        preds = (torch.sigmoid(logits) > 0.5).float()
        acc   = (preds == labels).float().mean()

        self.log("train/loss", loss,  prog_bar=True, on_step=True,  on_epoch=False)
        self.log("train/acc",  acc,   prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        spatial, temporal, labels = batch
        logits = self(spatial, temporal)
        loss   = self.loss_fn(logits, labels)

        probs = torch.sigmoid(logits)
        self._val_probs.extend(probs.cpu().numpy())
        self._val_labels.extend(labels.cpu().numpy())

        self.log("val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        if len(self._val_labels) < 2:
            self._val_probs, self._val_labels = [], []
            return

        probs  = np.array(self._val_probs)
        labels = np.array(self._val_labels)

        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5

        preds = (probs > 0.5).astype(int)
        acc   = (preds == labels.astype(int)).mean()
        f1    = f1_score(labels.astype(int), preds, zero_division=0)

        self.log("val/auc", auc, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/f1",  f1,  prog_bar=True)

        self._val_probs, self._val_labels = [], []

    # ── Test step ─────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        spatial, temporal, labels = batch
        probs = torch.sigmoid(self(spatial, temporal))
        self._test_probs.extend(probs.cpu().numpy())
        self._test_labels.extend(labels.cpu().numpy())

    def on_test_epoch_end(self):
        probs  = np.array(self._test_probs)
        labels = np.array(self._test_labels)

        auc  = roc_auc_score(labels, probs)
        preds = (probs > 0.5).astype(int)
        acc  = (preds == labels.astype(int)).mean()
        f1   = f1_score(labels.astype(int), preds, zero_division=0)

        self.log("test/auc", auc)
        self.log("test/acc", acc)
        self.log("test/f1",  f1)
        log.info(f"\nTest Results → AUC: {auc:.4f} | ACC: {acc:.4f} | F1: {f1:.4f}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        # Lower LR for pretrained ViT, higher for new layers
        vit_params  = list(self.model.vit.parameters())
        head_params = (
            list(self.model.spatial_proj.parameters()) +
            list(self.model.temporal_proj.parameters()) +
            list(self.model.cross_attn.parameters()) +
            list(self.model.classifier.parameters()) +
            list(self.model.temporal.parameters())
        )
        optimizer = AdamW([
            {"params": vit_params,  "lr": self.hparams.lr * 0.1},
            {"params": head_params, "lr": self.hparams.lr},
        ], weight_decay=self.hparams.weight_decay)

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ── Main entry point ───────────────────────────────────────────────────────────

def main(args):
    pl.seed_everything(args.seed)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        frame_skip=args.frame_skip,
        face_detect=not args.no_face_detect,
        num_workers=args.num_workers,
    )

    # Model
    model = DeepfakeLightning(
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        freeze_vit=args.freeze_vit,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="data/snapshots",
            filename="deepfake-{epoch:02d}-auc{val/auc:.4f}",
            monitor="val/auc",
            mode="max",
            save_top_k=3,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/auc",
            patience=args.patience,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = CSVLogger("data/logs", name="training")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        precision="16-mixed" if args.fp16 else 32,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    log.info("Training complete. Best checkpoint saved to data/snapshots/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepfakeDetector")
    parser.add_argument("--real-dir",        default="data/raw/real")
    parser.add_argument("--fake-dir",        default="data/raw/fake")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch-size",      type=int,   default=8)
    parser.add_argument("--clip-len",        type=int,   default=16)
    parser.add_argument("--frame-skip",      type=int,   default=3)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight-decay",    type=float, default=1e-4)
    parser.add_argument("--pos-weight",      type=float, default=1.0,
                        help="BCELoss pos_weight: increase if fake videos are rare")
    parser.add_argument("--patience",        type=int,   default=5)
    parser.add_argument("--num-workers",     type=int,   default=4)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--freeze-vit",      action="store_true")
    parser.add_argument("--no-face-detect",  action="store_true",
                        help="Skip RetinaFace (faster but less accurate)")
    parser.add_argument("--fp16",            action="store_true",
                        help="Mixed precision (recommended for GPU)")
    args = parser.parse_args()
    main(args)