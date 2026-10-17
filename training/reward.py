"""Rubric reward, per-run rubric modes, and the Rubric Dropout mask."""

import hashlib
import math
import random


def weighted_reward(rubric, verdicts, mask=None):
    """Weighted fraction of kept criteria satisfied. Verdicts may be fractional (averaged over repeats)."""
    if mask is None:
        mask = [1] * len(rubric)
    num = sum(m * c["weight"] * s for m, c, s in zip(mask, rubric, verdicts))
    den = sum(m * c["weight"] for m, c in zip(mask, rubric))
    return num / den


def run_rubric(example, mode, personalization_weight=1.0):
    """The rubric a run trains on, from an example's stored rubric.

    natural: as generated.
    fixed:   personalization criteria removed on memory-irrelevant queries (the targeted fix).
    personalization_weight multiplies personalization criteria's weights (positive control only).
    """
    rubric = example["rubric"]
    if mode == "fixed" and not example["relevant"]:
        rubric = [c for c in rubric if not c["personalization"]]
    elif mode != "natural" and mode != "fixed":
        raise ValueError(f"unknown rubric mode: {mode}")
    return [{**c, "weight": c["weight"] * (personalization_weight if c["personalization"] else 1.0)} for c in rubric]


def dropout_mask(n, prompt_id, step, frac, min_keep=3):
    """Drop floor(frac * n) criteria at random, keeping at least min_keep (or all, if n is smaller).

    Seeded by (prompt id, step): every rollout of a prompt in a step gets the same mask, so GRPO's
    within-group comparison stays valid, and the mask changes from step to step.
    """
    n_drop = min(math.floor(frac * n), max(n - min_keep, 0))
    seed = int.from_bytes(hashlib.sha256(f"{prompt_id}:{step}".encode()).digest()[:8], "big")
    dropped = set(random.Random(seed).sample(range(n), n_drop))
    return [0 if i in dropped else 1 for i in range(n)]
