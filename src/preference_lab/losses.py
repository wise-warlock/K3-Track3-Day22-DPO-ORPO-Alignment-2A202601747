from typing import cast

import numpy as np


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(x)) = -logaddexp(0, -x)."""
    return cast(np.ndarray, -np.logaddexp(0.0, -x))


def _log_odds(logp: np.ndarray) -> np.ndarray:
    """Compute log(p / (1 - p)) from log(p) safely."""
    # Clip logp to prevent numerical instability at 0.0 or extreme negative values
    clipped = np.clip(logp, -30.0, -1e-7)
    return cast(np.ndarray, clipped - np.log1p(-np.exp(clipped)))


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities."""
    policy_diff = policy_chosen_logps - policy_rejected_logps
    ref_diff = ref_chosen_logps - ref_rejected_logps
    margin = beta * (policy_diff - ref_diff)
    losses = -_log_sigmoid(margin)
    return float(np.mean(losses))


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute SFT loss + odds-ratio preference penalty."""
    chosen_log_odds = _log_odds(chosen_logps)
    rejected_log_odds = _log_odds(rejected_logps)
    log_odds_diff = chosen_log_odds - rejected_log_odds
    pref_loss = -lambda_orpo * np.mean(_log_sigmoid(log_odds_diff))
    total_loss = np.mean(sft_nll) + pref_loss
    return float(total_loss)
