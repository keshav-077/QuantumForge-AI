"""CryoForge RL Tiebreaker Package — Trajectory Collection, QLoRA SFT & Small-Model Evaluation.

Implements a strict zero-leakage two-split post-training and evaluation pipeline:
- Training & Experience Distillation Split: `EVAL_SEEDS` (`S01`–`S05` ONLY)
- Held-Out Unseen Evaluation Split: `TIEBREAKER_SEEDS` (`S06`, `S07`, `S08` ONLY)

Modules:
- `tiebreaker.collect_trajectories`: Collects/rolls out `S01`–`S05` expert and contrastive failure
  trajectories, exporting TRL multi-turn tool-calling SFT data (`sft_train_messages.jsonl`),
  DPO/GRPO contrast pairs (`preference_pairs.jsonl`), and an automatically distilled
  empirical policy (`experience_playbook.json`).
- `tiebreaker.train_sft`: Trains a QLoRA adapter (`Qwen/Qwen2.5-3B-Instruct` default) locally
  (`--local-gpu`), on Modal L4 cloud GPUs (`--modal`), or compiles the deterministic
  `S01`–`S05` experience policy (`--compile-experience`).
- `tiebreaker.eval_small_model`: Evaluates small open-weight / lightweight models
  (`gemma-4-26b-a4b-it`, `gemini-3.5-flash-lite`, `Qwen/Qwen2.5-3B-Instruct`) BEFORE vs AFTER
  training/experience on held-out seeds `S06`–`S08` over real MCP (`stdio` or `sse`).
"""

from __future__ import annotations

from pathlib import Path
from env_core.contracts import EVAL_SEEDS, TIEBREAKER_SEEDS

REPO_ROOT = Path(__file__).resolve().parent.parent
TIEBREAKER_RUNS_DIR = REPO_ROOT / "runs" / "tiebreaker"
TRAJECTORIES_DIR = REPO_ROOT / "runs" / "trajectories"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "cryoforge-qwen25-3b-lora"

ALLOWED_TRAIN_SEEDS: frozenset[str] = frozenset(EVAL_SEEDS)
FORBIDDEN_HOLDOUT_SEEDS: frozenset[str] = frozenset(TIEBREAKER_SEEDS)


def assert_train_seed_only(seed: str) -> None:
    """Raise ValueError if a held-out tiebreaker seed (`S06`–`S08`) enters training."""
    base_seed = seed.split("_")[0]
    if base_seed in FORBIDDEN_HOLDOUT_SEEDS or seed in FORBIDDEN_HOLDOUT_SEEDS:
        raise ValueError(
            f"DATA LEAKAGE BLOCKED: Held-out tiebreaker seed {seed!r} is strictly forbidden "
            f"in training/distillation (allowed base seeds: {sorted(ALLOWED_TRAIN_SEEDS)})."
        )
    if base_seed not in ALLOWED_TRAIN_SEEDS:
        raise ValueError(
            f"Unknown or non-training seed {seed!r}. Only {sorted(ALLOWED_TRAIN_SEEDS)} are permitted."
        )


__all__ = [
    "ALLOWED_TRAIN_SEEDS",
    "DEFAULT_CHECKPOINT_DIR",
    "FORBIDDEN_HOLDOUT_SEEDS",
    "REPO_ROOT",
    "TIEBREAKER_RUNS_DIR",
    "TRAJECTORIES_DIR",
    "assert_train_seed_only",
]
