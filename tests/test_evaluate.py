import pytest

from preference_lab.evaluate import pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0
    assert pairwise_accuracy(examples, [1.0], [2.0]) == 0.0


def test_pairwise_accuracy_ties() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a1", rejected="b1"),
        PreferenceExample(prompt="p2", chosen="a2", rejected="b2"),
    ]
    # One win (1.0), one tie (0.5) -> (1.0 + 0.5) / 2 = 0.75
    assert pairwise_accuracy(examples, [2.0, 1.5], [1.0, 1.5]) == 0.75


def test_pairwise_accuracy_length_mismatch() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError, match="Length mismatch"):
        pairwise_accuracy(examples, [1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="Length mismatch"):
        pairwise_accuracy(examples, [1.0], [1.0, 2.0])


def test_pairwise_accuracy_empty() -> None:
    assert pairwise_accuracy([], [], []) == 0.0


def test_preference_trainer(tmp_path: pytest.TempPathFactory) -> None:
    from preference_lab.trainers import PreferenceTrainer, TrainingConfig

    dpo_trainer = PreferenceTrainer(
        TrainingConfig(method="dpo", beta=0.1, output_dir=str(tmp_path), num_epochs=2)
    )
    dpo_res = dpo_trainer.train()
    assert dpo_res["method"] == "dpo"
    assert len(dpo_res["loss_history"]) == 2

    orpo_trainer = PreferenceTrainer(
        TrainingConfig(method="orpo", lambda_orpo=0.1, output_dir=str(tmp_path), num_epochs=2)
    )
    orpo_res = orpo_trainer.train()
    assert orpo_res["method"] == "orpo"
    assert len(orpo_res["loss_history"]) == 2
