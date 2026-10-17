"""The correctness checks from NOTES.md that a silent bug would invalidate.

    uv run --with pytest pytest tests
"""

from training.reward import dropout_mask, run_rubric, weighted_reward

RUBRIC = [
    {"criterion": "a", "weight": 5, "personalization": False},
    {"criterion": "b", "weight": 4, "personalization": False},
    {"criterion": "c", "weight": 3, "personalization": True},
    {"criterion": "d", "weight": 2, "personalization": False},
    {"criterion": "e", "weight": 1, "personalization": True},
]


def test_mask_shared_within_group_and_changes_across_steps():
    # Same prompt and step (the 8 rollouts of a group): identical mask.
    assert len({tuple(dropout_mask(10, "p001-q2", 7, 0.5)) for _ in range(8)}) == 1
    # Across steps the mask changes (at least sometimes).
    assert len({tuple(dropout_mask(10, "p001-q2", s, 0.5)) for s in range(20)}) > 1
    # Different prompts in the same step get independent masks.
    assert len({tuple(dropout_mask(10, f"p{i:03d}-q0", 7, 0.5)) for i in range(20)}) > 1


def test_mask_drops_floor_fraction_and_keeps_at_least_three():
    for n in range(1, 16):
        for step in range(5):
            kept = sum(dropout_mask(n, "x", step, 0.5))
            assert kept == max(n - n // 2, min(n, 3))
    assert sum(dropout_mask(10, "x", 0, 0.0)) == 10


def test_reward_uses_only_kept_criteria():
    verdicts = [1, 0, 1, 0, 1]
    assert weighted_reward(RUBRIC, verdicts) == (5 + 3 + 1) / 15
    mask = [1, 1, 0, 0, 1]  # keep a, b, e
    assert weighted_reward(RUBRIC, verdicts, mask) == (5 + 1) / (5 + 4 + 1)
    # A dropped criterion can't change the reward.
    assert weighted_reward(RUBRIC, [1, 0, 0, 1, 1], mask) == weighted_reward(RUBRIC, [1, 0, 1, 0, 1], mask)


def test_run_rubric_modes():
    irrelevant = {"rubric": RUBRIC, "relevant": False}
    relevant = {"rubric": RUBRIC, "relevant": True}
    assert run_rubric(irrelevant, "natural") == RUBRIC
    # Fixed removes personalization criteria only on irrelevant queries.
    assert [c["criterion"] for c in run_rubric(irrelevant, "fixed")] == ["a", "b", "d"]
    assert run_rubric(relevant, "fixed") == RUBRIC
    # Positive control scales only personalization criteria.
    weights = [c["weight"] for c in run_rubric(irrelevant, "natural", personalization_weight=3)]
    assert weights == [5, 4, 9, 2, 3]
