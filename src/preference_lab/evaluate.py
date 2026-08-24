from __future__ import annotations

import json
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return fraction where chosen score is greater than rejected score (ties count as 0.5)."""
    if not examples:
        return 0.0
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError(
            f"Length mismatch: examples ({len(examples)}), chosen_scores ({len(chosen_scores)}), "
            f"rejected_scores ({len(rejected_scores)})"
        )
    total_score = 0.0
    for c, r in zip(chosen_scores, rejected_scores):
        if c > r:
            total_score += 1.0
        elif c == r:
            total_score += 0.5
    return total_score / len(examples)


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
