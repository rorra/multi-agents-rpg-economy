"""Censored recovery-time analysis (design v1.0 s11.1, s11.4).

A recovery time that never occurred inside the horizon is censored: we know
only that it exceeds what we observed. Section 11.1 forbids replacing it with an
artificial value, so the ordinary paired Wilcoxon cannot be applied to these
observations as though they were measurements.

Tests, in order of priority (design s11.4):

1. The paired binary test is primary. The design pairs by seed, so each seed
   yields one A and one B observation. Whether each recovered within the horizon
   is a complete, uncensored fact, and McNemar's exact test on the discordant
   pairs uses it while respecting the pairing exactly.

2. A seed-stratified log-rank is reported alongside. SciPy 1.18 provides
   `scipy.stats.logrank` with no stratification parameter, and an unstratified
   test would discard the seed pairing, so the stratified statistic is
   implemented here. With one observation per group per stratum it has little
   power, which is why it is secondary.

3. Kaplan-Meier curves summarise each condition's recovery, using SciPy's
   `ecdf` with `CensoredData` so censored observations contribute exposure
   without contributing events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rpg_economy.metrics.run_metrics import Duration


@dataclass(frozen=True)
class Observation:
    """One seed's recovery observation under one condition."""

    seed: int
    duration: Duration

    @property
    def time(self) -> int:
        return self.duration.value

    @property
    def event(self) -> bool:
        """True if recovery was observed; False if the observation is censored."""
        return self.duration.observed


@dataclass(frozen=True)
class PairedBinaryResult:
    """McNemar's exact test on recovered-within-horizon (design v1.0 s11.4)."""

    n_pairs: int
    both_recovered: int
    neither_recovered: int
    only_left: int
    only_right: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.only_left + self.only_right

    def to_payload(self) -> dict[str, Any]:
        return {
            "test": "mcnemar_exact",
            "n_pairs": self.n_pairs,
            "both_recovered": self.both_recovered,
            "neither_recovered": self.neither_recovered,
            "only_left_recovered": self.only_left,
            "only_right_recovered": self.only_right,
            "discordant_pairs": self.discordant,
            "p_value": self.p_value,
        }


@dataclass(frozen=True)
class LogRankResult:
    """A seed-stratified log-rank test."""

    statistic: float
    p_value: float
    observed_left: float
    expected_left: float
    strata: int
    events: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "test": "seed_stratified_logrank",
            "statistic": self.statistic,
            "p_value": self.p_value,
            "observed_left": self.observed_left,
            "expected_left": self.expected_left,
            "strata": self.strata,
            "events": self.events,
        }


def censoring_rate(observations: list[Observation]) -> float | None:
    """Share of observations that never recovered (design v1.0 s11.1).

    Reported per condition and scenario, always, so that a favourable-looking
    median recovery time is read alongside how often recovery failed entirely.
    ``None`` with no observations: an empty group has no rate, and 0% would
    read as "every run recovered".
    """
    if not observations:
        return None
    return sum(1 for o in observations if o.duration.censored) / len(observations)


def paired_binary_test(left: list[Observation], right: list[Observation]) -> PairedBinaryResult:
    """McNemar's exact test on 'recovered within the horizon'.

    Only discordant pairs -- one recovered, the other did not -- carry
    information about a difference, which is what makes this the right paired
    test for an outcome that is fully observed even when the timing is not.
    """
    return mcnemar_exact(
        {o.seed: o.event for o in left},
        {o.seed: o.event for o in right},
    )


def mcnemar_exact(left: dict[int, bool], right: dict[int, bool]) -> PairedBinaryResult:
    """McNemar's exact test on any per-seed binary outcome.

    Factored out of :func:`paired_binary_test` because design s11.3 adds a
    second such outcome -- whether the shock-phase price was measurable at all
    -- which is paired the same way and must be tested the same way.
    """
    seeds = sorted(set(left) & set(right))

    both = neither = only_left = only_right = 0
    for seed in seeds:
        a, b = left[seed], right[seed]
        if a and b:
            both += 1
        elif not a and not b:
            neither += 1
        elif a:
            only_left += 1
        else:
            only_right += 1

    discordant = only_left + only_right
    if discordant == 0:
        p_value = 1.0
    else:
        # Exact binomial, two-sided, under H0 that a discordant pair falls
        # either way with equal probability.
        smaller = min(only_left, only_right)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)

    return PairedBinaryResult(
        n_pairs=len(seeds),
        both_recovered=both,
        neither_recovered=neither,
        only_left=only_left,
        only_right=only_right,
        p_value=p_value,
    )


def stratified_logrank(left: list[Observation], right: list[Observation]) -> LogRankResult:
    """Seed-stratified log-rank test.

    Implemented here because SciPy 1.18's ``logrank`` takes no strata, and an
    unstratified test would discard the seed pairing the trio design creates.

    Each seed is one stratum. Within a stratum, at every event time, the
    observed events in the left group are compared against their expectation
    under the null, using the hypergeometric variance; the contributions are
    then summed across strata. That is the standard construction, and it keeps
    each seed's comparison inside its own risk set.
    """
    by_seed: dict[int, tuple[list[Observation], list[Observation]]] = {}
    for observation in left:
        by_seed.setdefault(observation.seed, ([], []))[0].append(observation)
    for observation in right:
        by_seed.setdefault(observation.seed, ([], []))[1].append(observation)

    observed = expected = variance = 0.0
    events = 0
    strata = 0

    for _seed, (group_left, group_right) in sorted(by_seed.items()):
        if not group_left or not group_right:
            continue  # an unpaired seed carries no within-stratum comparison
        strata += 1
        times = sorted({o.time for o in group_left + group_right if o.event})
        for time in times:
            at_risk_left = sum(1 for o in group_left if o.time >= time)
            at_risk_right = sum(1 for o in group_right if o.time >= time)
            at_risk = at_risk_left + at_risk_right
            if at_risk < 2:
                continue
            d_left = sum(1 for o in group_left if o.event and o.time == time)
            d_total = d_left + sum(1 for o in group_right if o.event and o.time == time)
            if d_total == 0:
                continue

            events += d_total
            observed += d_left
            share = at_risk_left / at_risk
            expected += d_total * share
            variance += d_total * share * (1 - share) * (at_risk - d_total) / (at_risk - 1)

    if variance <= 0:
        return LogRankResult(
            statistic=0.0,
            p_value=1.0,
            observed_left=observed,
            expected_left=expected,
            strata=strata,
            events=events,
        )

    statistic = (observed - expected) / math.sqrt(variance)
    p_value = math.erfc(abs(statistic) / math.sqrt(2))
    return LogRankResult(
        statistic=statistic,
        p_value=p_value,
        observed_left=observed,
        expected_left=expected,
        strata=strata,
        events=events,
    )


def kaplan_meier(observations: list[Observation]) -> dict[str, Any]:
    """Kaplan-Meier survival estimate, censored observations respected."""
    from scipy.stats import CensoredData, ecdf

    if not observations:
        return {
            "times": [],
            "survival": [],
            "n": 0,
            "events": 0,
            "censored": 0,
            "censoring_rate": None,
            "median": None,
        }

    uncensored = [o.time for o in observations if o.event]
    right = [o.time for o in observations if not o.event]
    data = CensoredData(uncensored=uncensored, right=right)
    result = ecdf(data)

    times = [float(q) for q in result.sf.quantiles]
    survival = [float(p) for p in result.sf.probabilities]
    return {
        "times": times,
        "survival": survival,
        "n": len(observations),
        "events": len(uncensored),
        "censored": len(right),
        "censoring_rate": censoring_rate(observations),
        "median": km_median(times, survival),
    }


def km_median(times: list[float], survival: list[float]) -> float | None:
    """The Kaplan-Meier median: the first time the curve falls to 0.5 or below.

    ``None`` when it never does -- half the runs or more were still unrecovered
    when observation stopped, so the median lies beyond the data. The median of
    the *observed* times would ignore every censored run and flatter a
    condition that often failed to recover.
    """
    for time, probability in zip(times, survival, strict=True):
        if probability <= 0.5 + 1e-12:
            return time
    return None
