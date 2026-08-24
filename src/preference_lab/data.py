from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load preference examples from JSONL with line-numbered error reporting and duplicate checks."""
    examples: list[PreferenceExample] = []
    seen_prompts: set[str] = set()
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_no}: Invalid JSON - {exc}") from exc
            try:
                example = PreferenceExample.model_validate(payload)
            except (ValidationError, ValueError) as exc:
                raise ValueError(f"{file_path}:{line_no}: Invalid schema - {exc}") from exc

            norm_prompt = " ".join(example.prompt.strip().lower().split())
            if norm_prompt in seen_prompts:
                raise ValueError(
                    f"{file_path}:{line_no}: Duplicate prompt detected: {example.prompt}"
                )
            seen_prompts.add(norm_prompt)
            examples.append(example)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage."""
    if not examples:
        return [], []
    prompt_to_examples: dict[str, list[PreferenceExample]] = {}
    for ex in examples:
        prompt_to_examples.setdefault(ex.prompt, []).append(ex)

    prompts = list(prompt_to_examples.keys())
    rng = random.Random(seed)
    rng.shuffle(prompts)

    num_val_prompts = int(len(prompts) * validation_ratio)
    val_prompts = set(prompts[:num_val_prompts])

    train: list[PreferenceExample] = []
    val: list[PreferenceExample] = []
    for prompt, group in prompt_to_examples.items():
        if prompt in val_prompts:
            val.extend(group)
        else:
            train.extend(group)

    return train, val
