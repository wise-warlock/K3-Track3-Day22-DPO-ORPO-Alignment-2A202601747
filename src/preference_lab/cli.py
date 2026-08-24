from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import pairwise_accuracy, write_metrics

app = typer.Typer(help="Preference alignment lab CLI")

_STOPWORDS = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "of",
    "to",
    "for",
    "is",
    "are",
    "what",
    "how",
    "why",
    "and",
    "between",
    "or",
    "does",
    "do",
    "explain",
    "describe",
    "differentiate",
}


def score_response(prompt: str, response: str) -> float:
    """Deterministic heuristic scorer for evaluating responses on CPU."""
    p_words = set(re.findall(r"\w+", prompt.lower()))
    r_words = re.findall(r"\w+", response.lower())
    if not r_words:
        return 0.0

    content_p_words = {w for w in p_words if w not in _STOPWORDS and len(w) > 2}
    overlap = sum(1 for w in content_p_words if w in r_words)
    keyword_score = overlap / max(1, len(content_p_words))

    content_r_words = {w for w in r_words if w not in _STOPWORDS and len(w) > 2}
    content_density = len(content_r_words) / max(1, len(r_words))

    length = len(r_words)
    if length < 14:
        length_score = (length / 14.0) * 0.7
    elif length <= 45:
        length_score = 1.0
    else:
        length_score = max(0.5, 1.0 - (length - 45) * 0.01)

    return round(float(0.40 * keyword_score + 0.35 * content_density + 0.25 * length_score), 4)


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def evaluate(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config YAML file"),
    ] = Path("configs/local.yaml"),
) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    chosen_scores = [score_response(ex.prompt, ex.chosen) for ex in examples]
    rejected_scores = [score_response(ex.prompt, ex.rejected) for ex in examples]
    metrics = {"pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores)}
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


if __name__ == "__main__":
    app()
