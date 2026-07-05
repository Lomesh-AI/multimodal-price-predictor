"""Training loop. Handles checkpointing, early stopping on validation SMAPE,
and guards against the two most common silent failure modes: NaN loss and
CUDA OOM mid-epoch."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import build_loss
from src.training.scheduler import build_optimizer_and_scheduler
from src.utils.exceptions import CheckpointError, TrainingError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        training_cfg: Dict[str, Any],
        checkpoint_dir: str,
        device: Optional[str] = None,
    ):
        self.model = model
        self.cfg = training_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = build_loss(training_cfg["loss"])
        self.epochs = training_cfg["epochs"]
        self.patience = training_cfg.get("early_stopping_patience", 4)

        self.best_val_loss = math.inf
        self.epochs_without_improvement = 0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        total_steps = len(train_loader) * self.epochs
        optimizer, scheduler = build_optimizer_and_scheduler(self.model, self.cfg, total_steps)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_one_epoch(train_loader, optimizer, scheduler)
            val_loss = self._validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            logger.info(
                "Epoch %d/%d — train_smape=%.4f val_smape=%.4f",
                epoch, self.epochs, train_loss, val_loss,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint("best.pt", epoch, val_loss)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, self.patience,
                    )
                    break

        self._save_history(history)
        return history

    def _train_one_epoch(self, loader: DataLoader, optimizer, scheduler) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for batch_idx, batch in enumerate(loader):
            try:
                text_emb, image_emb, price = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                pred = self.model(text_emb, image_emb)
                loss = self.loss_fn(pred, price)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise TrainingError(
                        f"Non-finite loss at batch {batch_idx} (epoch step) — "
                        "check for zero-price rows or an exploding learning rate"
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                n_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("CUDA OOM at batch %d — reduce training.batch_size", batch_idx)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise TrainingError(
                        f"Out of memory during training at batch {batch_idx}. "
                        "Reduce batch_size in the config and retry."
                    ) from e
                raise

        if n_batches == 0:
            raise TrainingError("Training epoch had zero batches — check the DataLoader")
        return total_loss / n_batches

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            text_emb, image_emb, price = [t.to(self.device) for t in batch]
            pred = self.model(text_emb, image_emb)
            loss = self.loss_fn(pred, price)
            total_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            raise TrainingError("Validation set produced zero batches — check the split")
        return total_loss / n_batches

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        path = self.checkpoint_dir / filename
        try:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                path,
            )
        except OSError as e:
            raise CheckpointError(f"Failed to save checkpoint to {path}: {e}") from e
        logger.info("Saved checkpoint: %s (val_loss=%.4f)", path, val_loss)

    def _save_history(self, history: Dict[str, Any]) -> None:
        path = self.checkpoint_dir / "history.json"
        try:
            with path.open("w") as f:
                json.dump(history, f, indent=2)
        except OSError as e:
            logger.warning("Failed to save training history to %s: %s", path, e)
