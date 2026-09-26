"""Paired contrasts, survival analysis and per-scenario results (design v1.0 s11.3, s11.4)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from rpg_economy.analysis.contrasts import (
    contrasts_for_metric,
    holm_correct,
    paired_bootstrap_ci,
    paired_contrast,
    rank_biserial,
)
from rpg_economy.analysis.results import CONTINUOUS_METRICS, Results, build_results
from rpg_economy.analysis.survival import (
    Observation,
    censoring_rate,
    kaplan_meier,
    paired_binary_test,
    stratified_logrank,
)
from rpg_economy.i18n import clear_cache
from rpg_economy.metrics.run_metrics import Duration, RunMetrics

SEEDS = tuple(range(1, 13))


@pytest.fixture(autouse=True)
def _reset_locale():
    clear_cache()
    yield
    clear_cache()


def observations(times: list[tuple[int, int, bool]]) -> list[Observation]:
    return [Observation(seed=s, duration=Duration(v, c)) for s, v, c in times]


def fake_run(condition: str, seed: int, *, volatility: float, censored: bool = False) -> RunMetrics:
    return RunMetrics(
        run_id=f"{condition}{seed}",
        condition=condition,
        scenario="resource_scarcity",
        experiment="experiment_1",
        seed=seed,
        target_good="iron",
        target_kind="price",
        volatility={"base": 1.0, "shock": volatility, "recovery": 1.0},
        volatility_phase={"base": 1.0, "shock": volatility, "recovery": 1.0},
        volatility_windows={
            phase: {"windows": 4, "measurable": 4, "thin": 0, "trades": 20}
            for phase in ("base", "shock", "recovery")
        },
        stockouts={
            "good": "iron",
            "episodes": 1,
            "total_ticks": 5,
            "longest": 5,
            "by_phase": {"base": 5, "shock": 0, "recovery": 0},
        },
        volume_units={"base": 10, "shock": 5, "recovery": 8},
        volume_value={"base": 100.0, "shock": 50.0, "recovery": 80.0},
        recovery=Duration(20 + seed, censored),
        recovery_common_horizon=None,
        useful_production={
            "produced": 10,
            "useful": 6,
            "overproduction": 4,
            "useful_fraction": 0.6,
            "window_t": 20,
        },
        gini=0.3,
        coordination={
            "activations": 100,
            "wasted_activations": 40,
            "coordination_efficiency": 0.6,
            "activation_requests": 100,
            "activations_deferred": 0,
            "messages_sent": 3,
            "director_calls": 0,
            "director_errors": 0,
            "invalid_outputs": 0,
            "llm_calls": 100,
        },
        unanswered_requests=1,
        tokens={"by_bucket": {}, "authoritative_total": 1000, "budget": {}},
        exhaustion_tick=None,
        primary_metrics=("volatility", "stockouts", "recovery_time"),
    )


def sample_results() -> Results:
    runs = []
    for seed in SEEDS:
        runs.append(fake_run("A", seed, volatility=1.0 + seed * 0.01))
        runs.append(fake_run("B", seed, volatility=2.0 + seed * 0.01, censored=seed % 3 == 0))
        runs.append(fake_run("C", seed, volatility=1.5 + seed * 0.01))
    return build_results(runs)


# ---------------------------------------------------------------------------
# Effect sizes and intervals
# ---------------------------------------------------------------------------
def test_rank_biserial_is_zero_without_a_tendency() -> None:
    assert rank_biserial([1.0, -1.0, 2.0, -2.0]) == pytest.approx(0.0)


def test_rank_biserial_saturates_when_every_pair_agrees() -> None:
    assert rank_biserial([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert rank_biserial([-1.0, -2.0, -3.0]) == pytest.approx(-1.0)


def test_rank_biserial_ignores_zero_differences() -> None:
    assert rank_biserial([0.0, 0.0, 5.0]) == pytest.approx(1.0)


def test_the_bootstrap_interval_is_reproducible() -> None:
    """An interval that moved between analyses would not be a result."""
    differences = [float(x) for x in range(-5, 6)]
    assert paired_bootstrap_ci(differences, seed=7) == paired_bootstrap_ci(differences, seed=7)


def test_the_bootstrap_interval_brackets_the_median() -> None:
    differences = [2.0 + 0.1 * i for i in range(20)]
    low, high = paired_bootstrap_ci(differences, seed=1)
    assert low <= 2.95 <= high


# ---------------------------------------------------------------------------
# Pairing and correction
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_contrasts_are_paired_on_the_seed() -> None:
    """Pairing removes between-world variance; discarding it wastes the design."""
    left = {seed: float(seed) for seed in SEEDS}
    right = {seed: float(seed) + 1.0 for seed in SEEDS}
    contrast = paired_contrast("m", "A", "B", left, right)

    assert contrast.n_pairs == len(SEEDS)
    assert contrast.median_difference == pytest.approx(-1.0)
    assert contrast.rank_biserial == pytest.approx(-1.0)


def test_only_shared_seeds_are_paired() -> None:
    contrast = paired_contrast("m", "A", "B", {1: 1.0, 2: 2.0, 3: 3.0}, {2: 5.0, 3: 6.0})
    assert contrast.n_pairs == 2


def test_holm_correction_is_monotone_and_never_lowers_a_p_value() -> None:
    contrasts = [
        paired_contrast("m", "A", "B", dict.fromkeys(SEEDS, 1.0), dict.fromkeys(SEEDS, 2.0)),
        paired_contrast("m", "A", "C", dict.fromkeys(SEEDS, 1.0), dict.fromkeys(SEEDS, 1.5)),
        paired_contrast("m", "B", "C", dict.fromkeys(SEEDS, 2.0), dict.fromkeys(SEEDS, 1.5)),
    ]
    corrected = holm_correct(contrasts)
    for original, adjusted in zip(contrasts, corrected, strict=True):
        assert adjusted.p_value_holm >= original.p_value

    ordered = sorted(corrected, key=lambda c: c.p_value)
    values = [c.p_value_holm for c in ordered]
    assert values == sorted(values), "Holm-adjusted values must be monotone"


def test_a_contrast_family_is_one_metric_within_one_scenario() -> None:
    per_condition = {
        "A": dict.fromkeys(SEEDS, 1.0),
        "B": dict.fromkeys(SEEDS, 2.0),
        "C": dict.fromkeys(SEEDS, 1.5),
    }
    contrasts = contrasts_for_metric("volatility_shock", per_condition)
    assert {(c.left, c.right) for c in contrasts} == {("A", "B"), ("A", "C"), ("B", "C")}
    assert all(c.p_value_holm is not None for c in contrasts)


def test_few_pairs_are_flagged() -> None:
    contrast = paired_contrast("m", "A", "B", {1: 1.0, 2: 2.0}, {1: 2.0, 2: 3.0})
    assert contrast.note == "few_pairs"


# ---------------------------------------------------------------------------
# Censored recovery analysis
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_recovery_time_is_never_compared_with_the_paired_wilcoxon() -> None:
    """Design v1.0 s11.1: a censored observation is not a measurement."""
    assert "recovery" not in CONTINUOUS_METRICS
    assert "recovery_time" not in CONTINUOUS_METRICS


def test_censoring_rate_is_reported_per_condition() -> None:
    results = sample_results()
    assert set(results.survival["censoring_rate"]) == {"A", "B", "C"}
    assert results.survival["censoring_rate"]["B"] > 0
    assert results.survival["censoring_rate"]["A"] == 0


def test_the_paired_binary_test_uses_only_discordant_pairs() -> None:
    left = observations([(1, 10, False), (2, 10, False), (3, 10, True)])
    right = observations([(1, 10, False), (2, 10, True), (3, 10, True)])
    result = paired_binary_test(left, right)

    assert result.n_pairs == 3
    assert result.both_recovered == 1
    assert result.neither_recovered == 1
    assert result.discordant == 1
    assert result.only_left == 1


def test_the_paired_binary_test_is_neutral_without_discordant_pairs() -> None:
    same = observations([(s, 10, False) for s in range(1, 6)])
    assert paired_binary_test(same, same).p_value == 1.0


@pytest.mark.invariant
def test_the_logrank_is_stratified_by_seed() -> None:
    """SciPy 1.18 offers no strata, so the stratified statistic is implemented
    here rather than silently discarding the pairing."""
    left = observations([(s, 10, False) for s in range(1, 9)])
    right = observations([(s, 90, False) for s in range(1, 9)])
    result = stratified_logrank(left, right)

    assert result.strata == 8
    assert result.statistic > 0  # the left group recovers sooner
    assert result.p_value < 0.05


def test_the_logrank_ignores_unpaired_seeds() -> None:
    left = observations([(1, 10, False), (2, 10, False)])
    right = observations([(1, 50, False)])
    assert stratified_logrank(left, right).strata == 1


def test_kaplan_meier_respects_censoring() -> None:
    curve = kaplan_meier(observations([(1, 10, False), (2, 20, True), (3, 30, False)]))
    assert curve["n"] == 3
    assert curve["events"] == 2
    assert curve["censored"] == 1
    assert curve["censoring_rate"] == pytest.approx(1 / 3)


def test_censoring_rate_of_nothing_is_missing() -> None:
    """An empty group has no rate; 0% would read as "every run recovered"."""
    assert censoring_rate([]) is None


# ---------------------------------------------------------------------------
# The results file
# ---------------------------------------------------------------------------
def test_results_refuse_to_mix_scenarios() -> None:
    """Design v1.0 s11.4: contrasts are per scenario, never pooled."""
    runs = [fake_run("A", 1, volatility=1.0), fake_run("B", 2, volatility=1.0)]
    runs[1] = RunMetrics(**{**vars(runs[1]), "scenario": "demand_surge"})
    with pytest.raises(ValueError, match="several scenarios"):
        build_results(runs)


@pytest.mark.invariant
def test_the_results_file_is_language_neutral(tmp_path: Path) -> None:
    """Stored data holds enum values and numbers; language is applied later."""
    path = sample_results().write(tmp_path / "results.json")
    text = path.read_text(encoding="utf-8")

    for rendered in ("Escasez", "Resource scarcity", "Centralized", "Centralizada"):
        assert rendered not in text
    payload = json.loads(text)
    assert payload["scenario"] == "resource_scarcity"
    assert payload["conditions"] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Missing values drop the pair; they never enter a test as zero (design s11.3)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_an_unmeasurable_metric_drops_the_seed_pair() -> None:
    """A paired test compares differences; there is no difference to a blank.

    If a missing value were read as 0.0, the pair would still enter the Wilcoxon
    and would push the difference toward whichever condition failed to trade --
    manufacturing an effect out of an absence of data (design s11.3).

    Exercised on Gini, which has no measurability rule of its own: the
    selection-sensitive volatility metrics are suppressed outright instead.
    """
    runs = []
    for seed in range(1, 11):
        a = fake_run("A", seed, volatility=2.0)
        b = fake_run("B", seed, volatility=1.0)
        if seed <= 3:
            b.gini = None  # no holdings to value in these runs
        runs += [a, b]

    results = build_results(runs)
    table = results.by_condition("gini")

    assert sorted(table["B"]) == list(range(4, 11))
    assert all(value is not None for value in table["B"].values())

    contrast = next(c for c in results.contrasts["gini"] if c["right"] == "B")
    assert contrast["n_pairs"] == 7
    assert contrast["n_dropped"] == 3
    assert "dropped_unmeasured_pairs" in contrast["note"]


# ---------------------------------------------------------------------------
# Missingness caused by the treatment is not missing data (design s11.3)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_the_window_volatility_contrast_is_suppressed_when_measurability_differs() -> None:
    """Restricting to surviving pairs conditions the contrast on the treatment.

    Market thinness is *caused by* the orchestration architecture, so the seeds
    where B remains measurable are the seeds where B's markets held up. A
    contrast on those alone answers "among the runs where B traded, how did B
    do", which is not the question the design asks.
    """
    runs = []
    for seed in range(1, 11):
        runs.append(fake_run("A", seed, volatility=2.0))
        b = fake_run("B", seed, volatility=1.0)
        if seed <= 4:
            b.volatility["shock"] = None
        runs.append(b)

    results = build_results(runs)

    assert "volatility_shock" not in results.contrasts
    suppressed = results.suppressed_contrasts["volatility_shock"]
    assert suppressed["reason"] == "measurability_differs_by_condition"
    assert suppressed["companion_metric"] == "volatility_shock_phase"


@pytest.mark.invariant
def test_the_phase_level_companion_still_carries_the_comparison() -> None:
    """Suppressing one definition must not leave the metric uncontrasted."""
    runs = []
    for seed in range(1, 11):
        runs.append(fake_run("A", seed, volatility=2.0 + seed * 0.01))
        b = fake_run("B", seed, volatility=1.0 + seed * 0.01)
        if seed <= 4:
            b.volatility["shock"] = None  # window-level only; phase-level survives
        runs.append(b)

    results = build_results(runs)

    contrast = next(c for c in results.contrasts["volatility_shock_phase"] if c["right"] == "B")
    assert contrast["n_pairs"] == 10
    assert contrast["n_dropped"] == 0


# ---------------------------------------------------------------------------
# Missing values and guards
# ---------------------------------------------------------------------------
def test_a_contrast_with_no_pairs_is_missing_and_left_out_of_holm() -> None:
    empty = paired_contrast("m", "A", "B", {1: 1.0}, {2: 2.0})
    assert empty.n_pairs == 0
    assert empty.p_value is None
    assert empty.median_difference is None
    assert empty.ci_low is None and empty.ci_high is None

    tested = paired_contrast("m", "A", "C", {s: 1.0 + s for s in SEEDS}, dict.fromkeys(SEEDS, 1.0))
    corrected = holm_correct([empty, tested])
    assert corrected[0].p_value_holm is None
    # A family of one real test: Holm leaves its p-value alone.
    assert corrected[1].p_value_holm == pytest.approx(tested.p_value)


def test_measurability_is_compared_seed_by_seed_not_by_count() -> None:
    """Equal counts on different seeds still select the surviving pairs."""
    runs = []
    for seed in SEEDS:
        a = fake_run("A", seed, volatility=1.0)
        b = fake_run("B", seed, volatility=2.0)
        c = fake_run("C", seed, volatility=1.5)
        if seed <= 2:
            a.volatility["shock"] = None
        if seed in (3, 4):
            b.volatility["shock"] = None
            c.volatility["shock"] = None
        runs += [a, b, c]
    results = build_results(runs)
    assert "volatility_shock" in results.suppressed_contrasts


def test_results_refuse_to_mix_experiments() -> None:
    a = fake_run("A", 1, volatility=1.0)
    b = fake_run("B", 1, volatility=1.0)
    b.experiment = "experiment_2"
    with pytest.raises(ValueError, match="experiments"):
        build_results([a, b])


def test_results_refuse_a_duplicate_run() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_results([fake_run("A", 1, volatility=1.0), fake_run("A", 1, volatility=2.0)])


def test_results_refuse_seeds_outside_the_final_set() -> None:
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in (1, 2)]
    with pytest.raises(ValueError, match="final set"):
        build_results(runs, final_seeds=[1])
    assert build_results(runs, final_seeds=[1, 2]).seeds == [1, 2]


def test_an_unmeasurable_recovery_is_missing_not_censored() -> None:
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        if run.condition == "B" and run.seed == 1:
            run.recovery = None
    results = build_results(runs)

    assert results.survival["unmeasurable"] == {"A": 0, "B": 1, "C": 0}
    assert results.survival["censoring_rate"]["B"] == 0.0
    assert results.survival["kaplan_meier"]["B"]["n"] == len(SEEDS) - 1


def test_a_recovery_measurable_in_one_condition_only_withholds_that_pair() -> None:
    """Measurability is a treatment effect: the pair's survival tests are withheld."""
    from rpg_economy.analysis.contrasts import holm

    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        if run.condition == "B" and run.seed == SEEDS[0]:
            run.recovery = None
    survival = build_results(runs).survival

    withheld = {(p["left"], p["right"]) for p in survival["suppressed_pairs"]}
    assert withheld == {("A", "B"), ("B", "C")}
    for family in ("paired_binary", "stratified_logrank"):
        rows = survival[family]
        assert [(r["left"], r["right"]) for r in rows] == [("A", "C")]
        # The withheld pairs stay in the Holm family: m = 3.
        assert rows[0]["p_value_holm"] == holm([rows[0]["p_value"]], withheld=2)[0]
    assert [(r["left"], r["right"]) for r in survival["measurability"]] == [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ]
    assert survival["measurability"][0]["discordant_pairs"] == 1


def test_holm_on_p_values_counts_withheld_tests() -> None:
    from rpg_economy.analysis.contrasts import holm

    assert holm([0.01]) == [0.01]
    assert holm([0.01], withheld=2) == [0.03]
    assert holm([0.01, None], withheld=1) == [0.02, None]


def test_a_run_with_no_target_good_has_no_stockout_value() -> None:
    from rpg_economy.analysis.results import metric_value

    run = fake_run("A", 1, volatility=1.0)
    run.stockouts = {**run.stockouts, "good": None, "total_ticks": 0}
    assert metric_value(run, "stockout_ticks") is None


def test_missing_useful_production_and_efficiency_stay_missing() -> None:
    from rpg_economy.analysis.results import metric_value

    run = fake_run("A", 1, volatility=1.0)
    run.useful_production = {**run.useful_production, "useful_fraction": None}
    run.coordination = {**run.coordination, "coordination_efficiency": None}
    assert metric_value(run, "useful_production") is None
    assert metric_value(run, "coordination_efficiency") is None


def test_common_horizon_metrics_are_contrasted_when_present() -> None:
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        run.common_horizon = {"horizon": 30, "traded_value": 10.0 + run.seed, "stockout_ticks": 2}
    results = build_results(runs)
    assert "traded_value_common" in results.contrasts
    assert "stockout_ticks_common" in results.contrasts


# ---------------------------------------------------------------------------
# Holm over plain p-values, and the measurability family
# ---------------------------------------------------------------------------
def test_holm_on_plain_p_values() -> None:
    from rpg_economy.analysis.contrasts import holm

    adjusted = holm([0.01, 0.04, 0.03])
    # Sorted 0.01, 0.03, 0.04 -> x3, x2, x1, kept monotone.
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_never_lowers_a_p_value_and_caps_at_one() -> None:
    from rpg_economy.analysis.contrasts import holm

    raw = [0.2, 0.5, 0.001, 0.9]
    adjusted = holm(raw)
    assert all(a >= r for a, r in zip(adjusted, raw, strict=True))
    assert all(a <= 1.0 for a in adjusted)
    order = sorted(range(len(raw)), key=raw.__getitem__)
    ranked = [adjusted[i] for i in order]
    assert ranked == sorted(ranked)


def test_holm_leaves_missing_tests_out_of_the_family() -> None:
    from rpg_economy.analysis.contrasts import holm

    assert holm([None, 0.02, None]) == [None, pytest.approx(0.02), None]
    assert holm([]) == []


def test_holm_agrees_with_the_contrast_correction() -> None:
    from rpg_economy.analysis.contrasts import holm

    contrasts = [
        paired_contrast("m", "A", right, {s: 1.0 + s * k for s in SEEDS}, dict.fromkeys(SEEDS, 1.0))
        for right, k in (("B", 0.01), ("C", -0.02))
    ]
    corrected = holm_correct(contrasts)
    assert holm([c.p_value for c in contrasts]) == pytest.approx(
        [c.p_value_holm for c in corrected]
    )


@pytest.mark.invariant
def test_measurability_contrasts_are_holm_corrected_per_metric() -> None:
    from rpg_economy.analysis.contrasts import holm

    runs = []
    for seed in range(1, 11):
        runs.append(fake_run("A", seed, volatility=2.0))
        b = fake_run("B", seed, volatility=1.0)
        if seed <= 6:
            b.volatility["shock"] = None
            b.volatility_phase["shock"] = None
        runs.append(b)
        runs.append(fake_run("C", seed, volatility=1.5))

    results = build_results(runs)

    for metric in ("volatility_shock", "volatility_shock_phase"):
        rows = results.companion_outcomes[metric]["paired_binary"]
        assert {(r["left"], r["right"]) for r in rows} == {("A", "B"), ("A", "C"), ("B", "C")}
        for row in rows:
            assert row["p_value_holm"] >= row["p_value"]
        tested = [r["p_value"] for r in rows]
        assert [r["p_value_holm"] for r in rows] == pytest.approx(holm(tested))
    # The common-horizon variant only exists where a common horizon was computed.
    assert "volatility_shock_phase_common" not in results.companion_outcomes


# ---------------------------------------------------------------------------
# Suppression is decided pair by pair (design s11.3)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_suppression_is_decided_per_contrast_pair() -> None:
    """A and B measured on the same seeds keep their contrast when only C differs."""
    runs = []
    for seed in SEEDS:
        a = fake_run("A", seed, volatility=1.0 + seed * 0.01)
        b = fake_run("B", seed, volatility=2.0 + seed * 0.01)
        c = fake_run("C", seed, volatility=1.5 + seed * 0.01)
        if seed <= 3:
            c.volatility["shock"] = None
        runs += [a, b, c]
    results = build_results(runs)

    kept = results.contrasts["volatility_shock"]
    assert [(c["left"], c["right"]) for c in kept] == [("A", "B")]
    assert kept[0]["n_pairs"] == len(SEEDS)
    # The suppressed pairs are withheld, not untestable: they stay in the Holm
    # family, so A-B is corrected as one of three (design s11.4, user ruling).
    assert kept[0]["holm_family_size"] == 3
    assert kept[0]["p_value_holm"] == pytest.approx(min(1.0, 3 * kept[0]["p_value"]))

    suppressed = results.suppressed_contrasts["volatility_shock"]
    assert suppressed["pairs"] == [{"left": "A", "right": "C"}, {"left": "B", "right": "C"}]
    assert suppressed["companion_metric"] == "volatility_shock_phase"
    assert suppressed["holm_withheld"] == 2


def test_holm_counts_withheld_tests_in_the_family() -> None:
    from rpg_economy.analysis.contrasts import holm

    reported = [
        paired_contrast("m", "A", "B", {s: 1.0 + s for s in SEEDS}, dict.fromkeys(SEEDS, 1.0)),
        paired_contrast(
            "m", "A", "C", {s: 1.0 + s * 0.5 for s in SEEDS}, {s: 1.0 + (s % 3) for s in SEEDS}
        ),
    ]
    corrected = holm_correct(reported, withheld=1)
    ps = [c.p_value for c in reported]
    order = sorted(range(2), key=lambda i: ps[i])
    first = min(1.0, 3 * ps[order[0]])
    second = max(first, min(1.0, 2 * ps[order[1]]))
    assert corrected[order[0]].p_value_holm == pytest.approx(first)
    assert corrected[order[1]].p_value_holm == pytest.approx(second)
    # Without withheld tests it is plain Holm over the reported ones.
    assert [c.p_value_holm for c in holm_correct(reported)] == pytest.approx(holm(ps))


def test_a_withheld_contrast_without_measurable_pairs_leaves_the_family() -> None:
    """Only a contrast with no measurable pairs at all leaves the family (s11.4)."""
    runs = []
    for seed in SEEDS:
        a = fake_run("A", seed, volatility=1.0 + seed * 0.01)
        b = fake_run("B", seed, volatility=2.0 + seed * 0.01)
        c = fake_run("C", seed, volatility=1.5 + seed * 0.01)
        c.volatility["shock"] = None  # C never measurable: A-C and B-C have no pairs
        runs += [a, b, c]
    results = build_results(runs)

    assert results.suppressed_contrasts["volatility_shock"]["holm_withheld"] == 0
    (kept,) = results.contrasts["volatility_shock"]
    assert kept["holm_family_size"] == 1
    assert kept["p_value_holm"] == pytest.approx(kept["p_value"])


@pytest.mark.invariant
def test_the_phase_level_volatility_is_never_suppressed() -> None:
    """s11.3: the volatility comparison is sustained by the phase volatility."""
    runs = []
    for seed in SEEDS:
        a = fake_run("A", seed, volatility=1.0 + seed * 0.01)
        b = fake_run("B", seed, volatility=2.0 + seed * 0.01)
        c = fake_run("C", seed, volatility=1.5 + seed * 0.01)
        if seed <= 4:
            b.volatility_phase["shock"] = None
        runs += [a, b, c]
    results = build_results(runs)

    assert "volatility_shock_phase" not in results.suppressed_contrasts
    by_pair = {(c["left"], c["right"]): c for c in results.contrasts["volatility_shock_phase"]}
    assert set(by_pair) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert by_pair[("A", "B")]["n_pairs"] == len(SEEDS) - 4
    assert by_pair[("A", "B")]["n_dropped"] == 4
    assert by_pair[("A", "C")]["n_dropped"] == 0
    # Its measurability is still reported as an outcome.
    assert results.companion_outcomes["volatility_shock_phase"]["counts"]["B"] == {
        "measurable": len(SEEDS) - 4,
        "runs": len(SEEDS),
    }


def test_the_common_horizon_window_volatility_is_suppressed_per_pair() -> None:
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        window = None if (run.condition == "B" and run.seed <= 2) else 1.0 + run.seed * 0.1
        run.common_horizon = {
            "horizon": 30,
            "traded_value": 10.0 + run.seed,
            "stockout_ticks": 2,
            "volatility_shock": window,
            "volatility_shock_phase": 1.0 + run.seed * 0.1,
        }
    results = build_results(runs)

    suppressed = results.suppressed_contrasts["volatility_shock_common"]
    assert suppressed["companion_metric"] == "volatility_shock_phase_common"
    assert {(p["left"], p["right"]) for p in suppressed["pairs"]} == {("A", "B"), ("B", "C")}
    assert [(c["left"], c["right"]) for c in results.contrasts["volatility_shock_common"]] == [
        ("A", "C")
    ]
    assert "volatility_shock_phase_common" in results.contrasts


# ---------------------------------------------------------------------------
# Metrics that do not exist for a scenario are skipped (design s11.3)
# ---------------------------------------------------------------------------
def _liquidity_run(condition: str, seed: int) -> RunMetrics:
    run = fake_run(condition, seed, volatility=1.0)
    run.scenario = "liquidity_loss"
    run.target_good = None
    run.target_kind = "traded_value"
    run.primary_metrics = ("traded_value", "recovery_time")
    run.volatility = dict.fromkeys(("base", "shock", "recovery"))
    run.volatility_phase = dict.fromkeys(("base", "shock", "recovery"))
    run.stockouts = {**run.stockouts, "good": None, "total_ticks": 0}
    run.volume_value = {"base": 100.0, "shock": 50.0 + seed, "recovery": 80.0}
    run.coordination = {**run.coordination, "activation_concentration": 0.1}
    return run


def test_liquidity_loss_has_no_target_good_contrasts_or_measurability_rows() -> None:
    results = build_results([_liquidity_run(c, s) for c in "ABC" for s in SEEDS])

    for metric in ("volatility_shock", "volatility_shock_phase", "stockout_ticks"):
        assert metric not in results.contrasts
        assert metric not in results.measurability
        assert metric not in results.suppressed_contrasts
        assert metric not in results.companion_outcomes
    assert "traded_value" in results.contrasts
    assert results.primary_metrics == ("traded_value",)


# ---------------------------------------------------------------------------
# Secondary analyses are carried into the results (design s11.1, s11.3)
# ---------------------------------------------------------------------------
def test_secondary_analyses_are_contrasted_and_summarised() -> None:
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    offset = {"A": 0.0, "B": 1.0, "C": 0.5}
    for run in runs:
        shift = offset[run.condition] + run.seed * 0.01
        run.stockouts_inventory = {
            "good": "iron",
            "total_ticks": 3,
            "by_phase": {"base": 0, "shock": 3, "recovery": 0},
        }
        run.multi_good = {"volatility": 1.0 + shift, "volume": 0.8 + shift, "per_good": {}}
        run.propagation = {
            "source": "iron",
            "good": "melee_weapon",
            "mean_price": {"base": 30.0, "shock": 35.0 + shift, "recovery": 31.0},
            "stockout_ticks": {"base": 0, "shock": 2, "recovery": 0},
        }
        run.propagation_recovery = Duration(5 + run.seed, run.condition == "B")
        run.volume_by_good = {
            "iron": {
                "base": {"units": 10, "value": 100.0},
                "shock": {"units": 4 + run.seed, "value": 60.0},
                "recovery": {"units": 8, "value": 80.0},
            }
        }
    results = build_results(runs)

    for metric in (
        "stockout_ticks_inventory",
        "volatility_all_goods",
        "volume_all_goods",
        "propagation_price_shock",
        "propagation_stockout_ticks",
    ):
        assert metric in results.contrasts, metric
    assert results.survival_propagation["censoring_rate"]["B"] == 1.0
    assert results.survival_propagation["recovered_fraction"]["A"] == 1.0
    assert results.volume_by_good["A"]["iron"]["shock"]["units"] == pytest.approx(
        4 + statistics.median(SEEDS)
    )
    payload = results.to_payload()
    assert payload["survival_propagation"] is not None
    assert payload["volume_by_good"]


def test_there_is_no_propagation_block_without_a_raw_material_target() -> None:
    results = sample_results()
    assert results.survival_propagation is None
    assert "propagation_price_shock" not in results.contrasts


# ---------------------------------------------------------------------------
# Empty groups and the Kaplan-Meier median
# ---------------------------------------------------------------------------
def test_an_empty_recovery_group_has_no_rate_or_fraction() -> None:
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        if run.condition == "B":
            run.recovery = None
    results = build_results(runs)

    assert results.survival["censoring_rate"]["B"] is None
    assert results.survival["recovered_fraction"]["B"] is None
    assert results.survival["kaplan_meier"]["B"]["median"] is None
    assert results.survival["recovered_fraction"]["A"] == 1.0


def test_the_kaplan_meier_median_counts_censored_runs() -> None:
    from rpg_economy.analysis.survival import km_median

    # Two recoveries at 10 and 20, three runs censored beyond: the curve falls
    # to 0.8 and 0.6 and never reaches one half.
    curve = kaplan_meier(
        observations([(1, 10, False), (2, 20, False), (3, 30, True), (4, 30, True), (5, 30, True)])
    )
    assert curve["median"] is None
    # The median of the observed times alone would have said 15.

    curve = kaplan_meier(observations([(1, 10, False), (2, 20, False), (3, 30, True)]))
    assert curve["median"] == pytest.approx(20.0)
    assert km_median([5.0, 9.0], [0.5, 0.0]) == 5.0
    assert km_median([], []) is None


def test_the_results_file_records_whether_pilot_seeds_were_included() -> None:
    results = sample_results()
    assert results.to_payload()["pilot_seeds_included"] is False
    results.pilot_seeds_included = True
    assert results.to_payload()["pilot_seeds_included"] is True


def test_stockouts_over_every_good_are_a_contrasted_secondary() -> None:
    """liquidity_loss has no target good, so stockouts enter as this aggregate (s11.3)."""
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    offset = {"A": 0.0, "B": 1.0, "C": 0.5}
    for run in runs:
        shift = offset[run.condition] + run.seed * 0.01
        run.multi_good = {
            "volatility": 1.0,
            "volume": 1.0,
            "stockouts": 1.2 + shift,
            "per_good": {},
        }
    results = build_results(runs)

    assert "stockouts_all_goods" in results.contrasts
    ab = next(c for c in results.contrasts["stockouts_all_goods"] if c["right"] == "B")
    assert ab["median_difference"] == pytest.approx(-1.0)


def test_results_without_the_stockout_aggregate_do_not_list_it() -> None:
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        run.multi_good = {"volatility": 1.0, "volume": 1.0, "per_good": {}}
    results = build_results(runs)
    assert "stockouts_all_goods" not in results.contrasts


# ---------------------------------------------------------------------------
# Propagated price and all-goods volatility: selection-sensitive too (ruling)
# ---------------------------------------------------------------------------
def _propagated(run: RunMetrics, shock_price: float | None) -> RunMetrics:
    run.propagation = {
        "source": "iron",
        "good": "melee_weapon",
        "mean_price": {"base": 30.0, "shock": shock_price, "recovery": 31.0},
        "stockout_ticks": {"base": 0, "shock": run.seed % 3, "recovery": 0},
    }
    return run


@pytest.mark.invariant
def test_the_propagated_price_contrast_is_suppressed_per_pair_when_measurability_differs() -> None:
    runs = []
    for seed in SEEDS:
        for condition, offset in (("A", 0.0), ("B", 1.0), ("C", 2.0)):
            price = None if (condition == "B" and seed <= 4) else 35.0 + offset + seed * 0.1
            runs.append(_propagated(fake_run(condition, seed, volatility=1.0), price))
    results = build_results(runs)

    suppressed = results.suppressed_contrasts["propagation_price_shock"]
    assert suppressed["reason"] == "measurability_differs_by_condition"
    assert suppressed["companion_metric"] is None
    assert {(p["left"], p["right"]) for p in suppressed["pairs"]} == {("A", "B"), ("B", "C")}
    (kept,) = results.contrasts["propagation_price_shock"]
    assert (kept["left"], kept["right"]) == ("A", "C")
    # Holm keeps the metric's three contrasts: the two withheld pairs count.
    assert suppressed["holm_withheld"] == 2
    assert kept["holm_family_size"] == 3
    # Its measurability is reported per condition as an outcome of its own.
    outcome = results.companion_outcomes["propagation_price_shock"]
    assert outcome["counts"]["B"] == {"measurable": len(SEEDS) - 4, "runs": len(SEEDS)}
    assert outcome["counts"]["A"] == {"measurable": len(SEEDS), "runs": len(SEEDS)}
    assert len(outcome["paired_binary"]) == 3


@pytest.mark.invariant
def test_the_all_goods_volatility_contrast_is_suppressed_per_pair() -> None:
    runs = []
    for seed in SEEDS:
        for condition in "ABC":
            run = fake_run(condition, seed, volatility=1.0)
            volatility = None if (condition == "C" and seed <= 3) else 1.0 + seed * 0.05
            run.multi_good = {"volatility": volatility, "volume": 0.8, "stockouts": 1.0}
            runs.append(run)
    results = build_results(runs)

    suppressed = results.suppressed_contrasts["volatility_all_goods"]
    assert suppressed["companion_metric"] is None
    assert {(p["left"], p["right"]) for p in suppressed["pairs"]} == {("A", "C"), ("B", "C")}
    (kept,) = results.contrasts["volatility_all_goods"]
    assert (kept["left"], kept["right"]) == ("A", "B")
    assert kept["holm_family_size"] == 3
    assert results.companion_outcomes["volatility_all_goods"]["fraction"]["C"] == pytest.approx(
        (len(SEEDS) - 3) / len(SEEDS)
    )
    # A secondary metric measurable everywhere keeps all three contrasts.
    assert "volume_all_goods" not in results.suppressed_contrasts
    assert len(results.contrasts["volume_all_goods"]) == 3


def test_the_common_horizon_variants_are_suppressed_too() -> None:
    runs = []
    for seed in SEEDS:
        for condition in "ABC":
            run = _propagated(fake_run(condition, seed, volatility=1.0), 35.0 + seed)
            run.multi_good = {"volatility": 1.0 + seed * 0.05, "volume": 0.8, "stockouts": 1.0}
            thin = condition == "A" and seed <= 2
            run.common_horizon = {
                "horizon": 30,
                "traded_value": 10.0 + seed,
                "stockout_ticks": 2,
                "volatility_shock": 1.0 + seed * 0.1,
                "volatility_shock_phase": 1.0 + seed * 0.1,
                "volatility_all_goods": None if thin else 1.1 + seed * 0.01,
                "propagation_price_shock": None if thin else 30.0 + seed,
            }
            runs.append(run)
    results = build_results(runs)

    for metric in ("volatility_all_goods_common", "propagation_price_shock_common"):
        suppressed = results.suppressed_contrasts[metric]
        assert {(p["left"], p["right"]) for p in suppressed["pairs"]} == {("A", "B"), ("A", "C")}
        (kept,) = results.contrasts[metric]
        assert (kept["left"], kept["right"]) == ("B", "C")
        assert kept["holm_family_size"] == 3
        assert results.companion_outcomes[metric]["counts"]["A"]["measurable"] == len(SEEDS) - 2


def test_equal_measurability_keeps_every_propagated_price_contrast() -> None:
    runs = [
        _propagated(fake_run(c, s, volatility=1.0), None if s <= 2 else 35.0 + s)
        for c in "ABC"
        for s in SEEDS
    ]
    results = build_results(runs)
    assert "propagation_price_shock" not in results.suppressed_contrasts
    assert len(results.contrasts["propagation_price_shock"]) == 3
    assert "propagation_price_shock" in results.companion_outcomes


def test_a_pair_with_no_measurable_seeds_is_not_tested() -> None:
    """Design s11.4: a contrast with no measurable pairs is not a test."""
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        if run.condition in "BC":
            run.recovery = None
    survival = build_results(runs).survival
    for family in ("paired_binary", "stratified_logrank"):
        assert survival[family] == []
    withheld = {(p["left"], p["right"]) for p in survival["suppressed_pairs"]}
    assert withheld == {("A", "B"), ("A", "C")}
    # No seed measurable in both: withheld, but not counted in Holm.
    assert all(not p["in_family"] for p in survival["suppressed_pairs"])
    # B-C: neither measurable anywhere, so no test, and said so.
    assert survival["no_pairs"] == [{"left": "B", "right": "C"}]


def test_a_withheld_recovery_pair_hides_its_medians_and_curves(tmp_path: Path) -> None:
    from rpg_economy.analysis import report
    from rpg_economy.i18n import t

    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        if run.condition == "B" and run.seed == SEEDS[0]:
            run.recovery = None
    results = build_results(runs)
    table = "\n".join(report._recovery_table(results, results.survival))
    # A-B and B-C are withheld, so all three conditions sit in a withheld pair.
    assert table.count(t("report.value.suppressed")) == 3


def test_unanswered_requests_are_contrasted_between_b_and_c_only() -> None:
    """A sends no peer messages: A-B and A-C would test a constant zero."""
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        run.unanswered_requests = 0 if run.condition == "A" else run.seed % 4
    contrasts = build_results(runs).contrasts["unanswered_requests"]
    assert [(c["left"], c["right"]) for c in contrasts] == [("B", "C")]
