from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .losses import dpo_loss, orpo_loss


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    output_dir: str = "outputs"
    num_epochs: int = 3


class PreferenceTrainer:
    """CPU and extensible interface for DPO/ORPO training implementations."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> dict[str, Any]:
        """Train or simulate policy alignment and save metrics to output_dir."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)
        loss_history: list[float] = []

        for _ in range(self.config.num_epochs):
            batch_size = self.config.batch_size
            if self.config.method == "orpo":
                sft_nll = rng.uniform(0.5, 2.0, size=(batch_size,))
                chosen_logps = -rng.uniform(0.1, 1.0, size=(batch_size,))
                rejected_logps = -rng.uniform(1.0, 3.0, size=(batch_size,))
                loss = orpo_loss(sft_nll, chosen_logps, rejected_logps, self.config.lambda_orpo)
            else:  # dpo / default
                policy_chosen = -rng.uniform(0.2, 0.8, size=(batch_size,))
                policy_rejected = -rng.uniform(1.0, 2.5, size=(batch_size,))
                ref_chosen = -rng.uniform(0.5, 1.0, size=(batch_size,))
                ref_rejected = -rng.uniform(0.8, 1.5, size=(batch_size,))
                loss = dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    ref_chosen,
                    ref_rejected,
                    self.config.beta,
                )
            loss_history.append(float(loss))

        results: dict[str, Any] = {
            "method": self.config.method,
            "epochs": self.config.num_epochs,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
        }
        history_file = out_dir / "train_history.json"
        history_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results
