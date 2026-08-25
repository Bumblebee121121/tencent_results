"""Validation-loss early stopping for Stage 5 training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class EarlyStoppingDecision:
    is_best: bool
    significant_improvement: bool
    should_stop: bool


class ValidationLossEarlyStopping:
    """Track the absolute best loss and a separate meaningful-improvement baseline."""

    def __init__(
        self,
        patience: int,
        min_delta: float,
        *,
        best_loss: float = float("inf"),
        best_epoch: int = 0,
        reference_loss: float | None = None,
        bad_epochs: int = 0,
    ) -> None:
        if patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        if min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float(best_loss)
        self.best_epoch = int(best_epoch)
        self.reference_loss = float(best_loss if reference_loss is None else reference_loss)
        self.bad_epochs = int(bad_epochs)

    def update(self, loss: float, epoch: int) -> EarlyStoppingDecision:
        loss = float(loss)
        if not isfinite(loss):
            raise ValueError(f"validation loss must be finite, found {loss}")

        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss
            self.best_epoch = int(epoch)

        improvement = self.reference_loss - loss
        significant = loss < self.reference_loss and improvement >= self.min_delta
        if significant:
            self.reference_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return EarlyStoppingDecision(
            is_best=is_best,
            significant_improvement=significant,
            should_stop=self.bad_epochs >= self.patience,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "reference_loss": self.reference_loss,
            "bad_epochs": self.bad_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }

    @classmethod
    def resume(
        cls,
        state: Mapping[str, Any] | None,
        *,
        patience: int,
        min_delta: float,
        checkpoint_loss: float,
        checkpoint_epoch: int,
    ) -> "ValidationLossEarlyStopping":
        state = state or {}
        return cls(
            patience,
            min_delta,
            best_loss=float(state.get("best_loss", checkpoint_loss)),
            best_epoch=int(state.get("best_epoch", checkpoint_epoch)),
            reference_loss=float(state.get("reference_loss", checkpoint_loss)),
            bad_epochs=int(state.get("bad_epochs", 0)),
        )
