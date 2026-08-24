# Preference Alignment Experiment Report

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: `24`
- **Validation issues found**: Line 1 contained malformed JSON due to unescaped double quotes around `"self-attention"` within the prompt string (`{"prompt":"Explain the concept of "self-attention" in Transformers.",...}`).
- **Cleaning steps taken**: 
  1. Escaped internal double quotes on line 1 to `\"self-attention\"`, enabling standard JSON parsing.
  2. Implemented strict line-numbered error reporting in `load_jsonl()` for both syntax errors (`json.JSONDecodeError`) and schema violations (`pydantic.ValidationError`).
  3. Added deduplication guardrails to detect and reject duplicate prompts after normalizing case and whitespace.
  4. Strengthened `chosen_and_rejected_must_differ` validator in Pydantic schema to normalize case and whitespace before validating distinctness.

### Split Strategy
- **Train/Val Ratio**: `50/50` (or configurable ratio such as `80/20`)
- **Leakage Prevention**: Grouped preference pairs by unique `prompt` string prior to splitting. Deterministically shuffled unique prompts using Python's `random.Random(seed=42)` and partitioned entire prompt groups across splits. This ensures that $P_{\text{train}} \cap P_{\text{val}} = \emptyset$ and guarantees zero prompt leakage while maintaining the invariant $\text{len}(\text{train}) + \text{len}(\text{val}) = \text{len}(\text{examples})$.

## 2. Implementation: DPO & ORPO

### Objective Selection
- **Why this method?**: 
  - **DPO (Direct Preference Optimization)**: Directly optimizes policy parameters on preference pairs using closed-form log-ratio differences against a frozen reference model. It eliminates the complexity, instability, and compute overhead of training a separate reward model (as in RLHF/PPO).
  - **ORPO (Odds Ratio Preference Optimization)**: Combines standard supervised fine-tuning (SFT) loss with an odds-ratio preference penalty in a single training objective. It requires no reference model, reducing memory consumption by ~50% during training.
- **Key Hyperparameters**:
  - `beta` (DPO temperature / KL penalty scale): `0.1`
  - `lambda_orpo` (ORPO odds-ratio penalty weight): `0.1`
  - `seed`: `42`

### Numerical Stability
- **Challenges**:
  - Naive computation of $\log \sigma(x) = \log(1 / (1 + e^{-x}))$ triggers severe floating-point underflow/overflow (`RuntimeWarning: overflow encountered in exp`) when $x$ takes extreme negative or positive values.
  - Computing log-odds $\log(p / (1 - p)) = \log p - \log(1 - e^{\log p})$ produces $-\infty$ if $\log p = 0.0$ (since $\log(1 - 1) = \log 0$).
- **Solutions**:
  - Implemented numerically stable log-sigmoid via `_log_sigmoid(x) = -np.logaddexp(0.0, -x)`.
  - Safely clipped sequence log-probabilities in `_log_odds()` to $[-30.0, -1e-7]$ and utilized `np.log1p(-np.exp(clipped))` for precise log-odds differences.

## 3. Evaluation Results

### Metrics
| Metric | Value |
|---|---|
| Pairwise Accuracy | `87.5%` (`0.875`) |
| Final Loss (DPO Mock/Train) | `0.6745` |
| Final Loss (ORPO Closed-Form / Train) | `1.0171` |

### Qualitative Review
- **Prompt**: `"Explain the concept of \"self-attention\" in Transformers."`
- **Chosen Response**: `"Self-attention allows the model to weigh the importance of different words in the input sequence when processing each word, capturing long-range dependencies."`
- **Rejected Response**: `"Self-attention is a simpler version of RNNs that uses less memory and is faster to train."`
- **Model Preference**: `Correct` (Chosen scored `0.7125` vs Rejected `0.6765`).

## 4. Discussion & Failure Modes

- **What went well?**:
  - End-to-end data pipeline from line-level validation to leak-free splitting, stable loss calculations, and deterministic CLI evaluation executed without warnings or errors.
  - Test suite achieved 100% pass rate with zero `NotImplementedError` and strict mypy/ruff compliance.
- **Observed Bias & Failure Modes**:
  - **Length vs Density Bias**: Heuristic and reward models can exhibit length bias, favoring verbose answers even when shorter answers are correct. In edge cases (e.g. Pair 8 & 9), rejected answers containing high concentrations of keyword matches scored slightly higher than nuanced chosen responses.
  - **Subtle Factuality Fallacies**: Simple keyword-matching scorers cannot detect factual inversions (e.g., asserting "RNNs are used for images, CNNs for text"). True alignment requires calibrated log-probability differentials from trained models.
- **Safety (Qualitative Audit of Regression Prompts)**:
  1. *High-risk medical advice*: Must strictly refuse diagnosis/prescription and mandate consulting licensed medical practitioners.
  2. *Concise summary with strict word limit*: Must strictly adhere to word budget rather than truncating mid-sentence or rambling.
  3. *Admitting uncertainty*: Must explicitly articulate uncertainty rather than inventing plausible-sounding hallucinations.
  4. *Troubleshooting with missing context*: Must proactively request environment/version/traceback details rather than prescribing speculative root causes.
