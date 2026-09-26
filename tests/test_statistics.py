"""Paired contrasts, survival analysis and per-scenario results (design v1.0 s11.3, s11.4)."""

from __future__ import annotations

import json
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
