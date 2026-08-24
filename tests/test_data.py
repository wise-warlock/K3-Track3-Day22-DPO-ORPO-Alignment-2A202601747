from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_error_message_includes_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt":"a","chosen":"b","rejected":"c"}\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match="2"):
        load_jsonl(bad)


def test_schema_error_includes_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad_schema.jsonl"
    bad.write_text(
        '{"prompt":"a","chosen":"b","rejected":"c"}\n{"prompt":"d","chosen":"same","rejected":"same"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="2"):
        load_jsonl(bad)


def test_duplicate_prompt_detected(tmp_path: Path) -> None:
    dup = tmp_path / "dup.jsonl"
    dup.write_text(
        '{"prompt":"What is AI?","chosen":"Good answer","rejected":"Bad answer"}\n'
        '{"prompt":" what is ai? ","chosen":"Another answer","rejected":"Yet another"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate prompt"):
        load_jsonl(dup)


def test_split_has_no_prompt_leakage() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5, seed=42)
    assert len(train) + len(val) == len(examples)
    train_prompts = {e.prompt for e in train}
    val_prompts = {e.prompt for e in val}
    assert not (train_prompts & val_prompts)


def test_split_empty_and_edge_cases() -> None:
    train, val = split_by_prompt([], validation_ratio=0.2)
    assert train == [] and val == []
