def weighted_reward(rubric, verdicts, mask=None):
    """Weighted fraction of kept criteria satisfied. Verdicts may be fractional (averaged over repeats)."""
    if mask is None:
        mask = [1] * len(rubric)
    num = sum(m * c["weight"] * s for m, c, s in zip(mask, rubric, verdicts))
    den = sum(m * c["weight"] for m, c in zip(mask, rubric))
    return num / den
