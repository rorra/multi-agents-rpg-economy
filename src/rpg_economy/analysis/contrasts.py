"""Statistical contrasts (design v1.0 s11.4).

Contrasts run per scenario; primary values from different shocks are never
pooled into one global test. The main contrast is A vs B in the primary
scenario, with A-C and B-C secondary.

Every test is paired on the seed, because a seed defines a trio sharing
agents, endowments, shock and world randomness. Pairing is what removes
between-world variance from the comparison, so discarding it would throw away
most of the design's power.

Reported for every contrast, not only p-values (s11.4): the paired effect size
as rank-biserial correlation, the median of paired differences, and a paired
bootstrap confidence interval. Holm correction is applied per metric within a
scenario.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE = 0.95


@dataclass(frozen=True)
class PairedContrast:
    """One paired comparison of two conditions on one metric."""

    metric: str
    left: str
    right: str
    n_pairs: int
    n_dropped: int
    median_left: float | None
    median_right: float | None
    median_difference: float | None
    rank_biserial: float | None
    p_value: float | None
    p_value_holm: float | None
    ci_low: float | None
    ci_high: float | None
    note: str = ""

    @property
    def testable(self) -> bool:
        """False when no seed pair carried a value in both conditions."""
        return self.n_pairs > 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "left": self.left,
            "right": self.right,
            "n_pairs": self.n_pairs,
            "n_dropped": self.n_dropped,
            "median_left": self.median_left,
            "median_right": self.median_right,
            "median_difference": self.median_difference,
            "rank_biserial": self.rank_biserial,
            "p_value": self.p_value,
            "p_value_holm": self.p_value_holm,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "note": self.note,
        }


def rank_biserial(differences: list[float]) -> float:
    """Matched-pairs rank-biserial correlation (design v1.0 s11.4).

    The signed-rank effect size: the difference between the proportion of
    summed ranks favouring each direction. Zero means no tendency; +/-1 means
    every pair moved the same way.
    """
    non_zero = [d for d in differences if d != 0]
    if not non_zero:
        return 0.0

    ranks = _average_ranks([abs(d) for d in non_zero])
    total = sum(ranks)
    if total == 0:
        return 0.0
    positive = sum(r for r, d in zip(ranks, non_zero, strict=True) if d > 0)
    negative = sum(r for r, d in zip(ranks, non_zero, strict=True) if d < 0)
    return (positive - negative) / total


def _average_ranks(values: list[float]) -> list[float]:
    """Ranks with ties averaged, so tied magnitudes cannot bias the effect size."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = average
        index = stop + 1
    return ranks


def paired_bootstrap_ci(
    differences: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the median paired difference.

    Resamples pairs, not observations, so the pairing survives the
    resampling. Seeded, so the reported interval is reproducible.
    """
    if not differences:
        return (0.0, 0.0)
    import random

    rng = random.Random(seed)
    n = len(differences)
    medians = []
    for _ in range(resamples):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    alpha = (1 - CONFIDENCE) / 2
    low = medians[int(alpha * resamples)]
    high = medians[min(resamples - 1, int((1 - alpha) * resamples))]
    return (low, high)


def wilcoxon_signed_rank(differences: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired differences."""
    non_zero = [d for d in differences if d != 0]
    if len(non_zero) < 1:
        return 1.0
    from scipy.stats import wilcoxon

    try:
        return float(wilcoxon(non_zero).pvalue)
    except ValueError:
        return 1.0


def paired_contrast(
    metric: str,
    left_name: str,
    right_name: str,
    left: dict[int, float],
    right: dict[int, float],
    *,
    bootstrap_seed: int = 0,
    seeds_run: tuple[int, ...] = (),
) -> PairedContrast:
    """Compare two conditions on one metric, paired by seed.

    ``left`` and ``right`` hold only seeds with a usable value, so a seed whose
    value is missing in either condition drops out of the pair entirely
    (design s11.3: missing, never zero) -- a paired test compares differences, and there is no
    difference to take when one side was never measured. ``seeds_run`` names the
    seeds both conditions actually executed, which is what makes the dropped
    count visible instead of silently shrinking n.
    """
    seeds = sorted(set(left) & set(right))
    differences = [left[seed] - right[seed] for seed in seeds]
    dropped = max(0, len(set(seeds_run)) - len(seeds)) if seeds_run else 0

    if not seeds:
        # Nothing to compare. Every statistic is missing rather than a
        # placeholder zero that would read as "no difference, p = 1".
        return PairedContrast(
            metric=metric,
            left=left_name,
            right=right_name,
            n_pairs=0,
            n_dropped=dropped,
            median_left=None,
            median_right=None,
            median_difference=None,
            rank_biserial=None,
            p_value=None,
            p_value_holm=None,
            ci_low=None,
            ci_high=None,
            note="no_pairs",
        )

    notes = []
    if len(seeds) < 6:
        notes.append("few_pairs")
    if dropped:
        notes.append("dropped_unmeasured_pairs")
    note = ";".join(notes)

    low, high = paired_bootstrap_ci(differences, seed=bootstrap_seed)
    return PairedContrast(
        metric=metric,
        left=left_name,
        right=right_name,
        n_pairs=len(seeds),
        n_dropped=dropped,
        median_left=statistics.median(left[s] for s in seeds),
        median_right=statistics.median(right[s] for s in seeds),
        median_difference=statistics.median(differences),
        rank_biserial=rank_biserial(differences),
        p_value=wilcoxon_signed_rank(differences),
        p_value_holm=None,
        ci_low=low,
        ci_high=high,
        note=note,
    )


def holm(p_values: list[float | None], *, withheld: int = 0) -> list[float | None]:
    """Holm-adjusted p-values for one family; ``None`` entries are not tests.

    ``withheld`` counts tests of the family suppressed by measurability: they
    stay in ``m`` and rank last, as in :func:`holm_correct`.
    """
    if withheld < 0:
        raise ValueError("withheld must be zero or more")
    tested = [i for i, p in enumerate(p_values) if p is not None]
    ordered = sorted(tested, key=lambda i: p_values[i])
    m = len(tested) + withheld
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return [adjusted.get(i) for i in range(len(p_values))]


def holm_correct(contrasts: list[PairedContrast], *, withheld: int = 0) -> list[PairedContrast]:
    """Holm-Bonferroni correction across a family of contrasts.

    Applied per metric within a scenario (design v1.0 s11.4): the family is
    the set of condition contrasts for one metric, so correcting across
    scenarios or across unrelated metrics would over-penalise.

    A contrast with no pairs has no p-value and is left out of the family: it
    is not a test, so it must not inflate the multiplier of the ones that are.

    ``withheld`` counts contrasts of the family that were deliberately not
    reported (suppressed by measurability, design s11.3). They stay in the
    family: withholding a result is a choice made after seeing the data, and
    dropping it from ``m`` would reward that choice with a smaller correction.
    Their p-values are unknown, so they rank last, where they cannot lower the
    multiplier of any reported contrast.
    """
    if withheld < 0:
        raise ValueError("withheld must be zero or more")
    tested = [i for i, contrast in enumerate(contrasts) if contrast.p_value is not None]
    ordered = sorted(tested, key=lambda i: contrasts[i].p_value)
    m = len(tested) + withheld
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, index in enumerate(ordered):
        value = min(1.0, (m - rank) * contrasts[index].p_value)
        running = max(running, value)  # Holm's adjusted values are monotone
        adjusted[index] = running

    return [
        PairedContrast(**{**vars(contrast), "p_value_holm": adjusted.get(i)})
        for i, contrast in enumerate(contrasts)
    ]


def contrasts_for_metric(
    metric: str,
    per_condition: dict[str, dict[int, float]],
    *,
    pairs: tuple[tuple[str, str], ...] = (("A", "B"), ("A", "C"), ("B", "C")),
    seeds_run: dict[str, set[int]] | None = None,
    bootstrap_seed: int = 0,
    withheld: int = 0,
) -> list[PairedContrast]:
    """Every condition contrast for one metric, Holm-corrected as one family.

    ``withheld`` is the number of the family's contrasts suppressed rather than
    computed; they still count in the family (see :func:`holm_correct`).
    """
    ran = seeds_run or {}
    computed = [
        paired_contrast(
            metric,
            left,
            right,
            per_condition.get(left, {}),
            per_condition.get(right, {}),
            bootstrap_seed=bootstrap_seed,
            seeds_run=tuple(sorted(ran.get(left, set()) & ran.get(right, set()))),
        )
        for left, right in pairs
        if left in per_condition and right in per_condition
    ]
    return holm_correct(computed, withheld=withheld)
