"""Campaign planning, pricing and execution (IMPLEMENTATION.md phase 7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpg_economy.campaign import build_plan, estimate_cost, measure_profile, scenario_tier
from rpg_economy.campaign.cost import CostError, RunTokens, TokenProfile
from rpg_economy.campaign.plan import (
    CampaignPhase,
    PlanError,
    PlannedTrio,
    ScenarioTier,
    experiments_for,
)
from rpg_economy.campaign.runner import run_campaign, trio_directory
from rpg_economy.cli import main
from rpg_economy.config.models import ModelPricing
from rpg_economy.domain import Condition, ExperimentRegime, Scenario
from tests.conftest import make_config


def campaign_config(**overrides):
    """A compressed run so a whole campaign fits in a test."""
    base = {
        "phases": {"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 4},
        "time": {"horizon_ticks": 12},
        "metrics": {"volatility_window_w": 2, "recovery_hold_ticks_m": 2},
        "seeds": {"pilot": [1, 2, 3, 4, 5, 6, 7, 8], "final": list(range(101, 116))},
        # The full hierarchy, so these tests exercise the whole matrix rather than
        # the narrower scope of design s10.
        "campaign": {
            "secondary_scenarios": ["demand_surge", "productivity_drop", "liquidity_loss"],
            "pilot_calibration_horizon": 8,
        },
    }
    base.update(overrides)
    return make_config(**base)


PILOT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]

PRICING = ModelPricing(
    input_per_1m_tokens="1.25",
    output_per_1m_tokens="10.00",
    currency="USD",
    quoted_on="2026-09-02",
)


# ---------------------------------------------------------------------------
# The pre-registered hierarchy (design v1.0 s10, s11.4)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_only_the_primary_scenario_runs_experiment_2() -> None:
    """Design v1.0 s11.4: secondary scenarios are Experiment 1 only."""
    assert experiments_for(Scenario.RESOURCE_SCARCITY) == (
        ExperimentRegime.EQUAL_ACTIVITY,
        ExperimentRegime.EQUAL_COST,
    )
    for scenario in (
        Scenario.DEMAND_SURGE,
        Scenario.PRODUCTIVITY_DROP,
        Scenario.LIQUIDITY_LOSS,
    ):
        assert experiments_for(scenario) == (ExperimentRegime.EQUAL_ACTIVITY,)


@pytest.mark.invariant
def test_the_pilot_is_experiment_1_only() -> None:
    """Experiment 2's budget is derived from the Experiment 1 pilot (s8.4).

    A pilot that ran Experiment 2 would need the number it exists to produce.
    """
    for scenario in Scenario:
        assert experiments_for(scenario, CampaignPhase.PILOT) == (ExperimentRegime.EQUAL_ACTIVITY,)


@pytest.mark.invariant
def test_noisy_info_is_never_planned_implicitly() -> None:
    """Design v1.0 s10: admissible only if defined before final runs."""
    config = campaign_config()
    plan = build_plan(config, phase="pilot")

    assert Scenario.NOISY_INFO not in plan.by_scenario()
    assert scenario_tier(Scenario.NOISY_INFO) is ScenarioTier.OPTIONAL

    explicit = build_plan(config, phase="pilot", scenarios=(Scenario.NOISY_INFO,))
    assert Scenario.NOISY_INFO in explicit.by_scenario()
    assert any("outside the pre-registered hierarchy" in w for w in explicit.warnings)


@pytest.mark.invariant
def test_a_final_campaign_below_the_seed_floor_is_refused() -> None:
    """Silently running eight seeds where ten are required would be invisible."""
    config = campaign_config(
        seeds={"pilot": PILOT_SEEDS, "final": [101, 102, 103]},
        budget={"experiment_2_total_tokens": 1_000_000},
    )
    with pytest.raises(PlanError, match="at least 10 final seeds"):
        build_plan(config, phase="final")


def test_a_reduced_campaign_runs_the_primary_scenario_under_experiment_1_only() -> None:
    """primary_only: a few seeds, no secondary scenarios, no Experiment 2 budget needed."""
    config = campaign_config(
        seeds={"pilot": PILOT_SEEDS, "final": [101]},
        campaign={
            "secondary_scenarios": ["demand_surge"],
            "pilot_calibration_horizon": 8,
            "primary_only": True,
        },
    )
    plan = build_plan(config, phase="final")
    assert plan.trios == [
        PlannedTrio(
            scenario=Scenario.RESOURCE_SCARCITY,
            experiment=ExperimentRegime.EQUAL_ACTIVITY,
            seed=101,
        )
    ]
    assert any("reduced campaign" in str(w) for w in plan.warnings)


def test_a_final_campaign_below_the_seed_target_warns_but_proceeds() -> None:
    config = campaign_config(
        seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 113))},
        budget={"experiment_2_total_tokens": 1_000_000},
    )
    plan = build_plan(config, phase="final")
    assert any("targets 15" in w for w in plan.warnings)


@pytest.mark.invariant
def test_experiment_2_without_a_derived_budget_fails_at_plan_time() -> None:
    """Not mid-batch, after the Experiment 1 trios have already been paid for."""
    config = campaign_config(seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))})
    with pytest.raises(PlanError, match="experiment_2_total_tokens"):
        build_plan(config, phase="final")


def test_the_plan_covers_every_seed_in_every_cell() -> None:
    config = campaign_config(
        seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))},
        budget={"experiment_2_total_tokens": 1_000_000},
    )
    plan = build_plan(config, phase="final")

    # primary x 2 experiments x 15 seeds, plus 3 secondary x 1 experiment x the
    # first 8 seeds (design s11.4)
    assert len(plan.trios) == 2 * 15 + 3 * 8
    assert plan.total_runs == len(plan.trios) * 3
    secondary = {t.seed for t in plan.trios if t.scenario is not Scenario.RESOURCE_SCARCITY}
    assert secondary == set(range(101, 109))


@pytest.mark.invariant
def test_the_adopted_final_campaign_is_96_runs() -> None:
    """Design s11.4: 12 x 3 x 2 = 72 primary runs, plus 8 x 3 = 24 secondary."""
    config = make_config(budget={"experiment_2_total_tokens": 1_000_000})
    plan = build_plan(config, phase="final")
    assert plan.total_runs == 96


# ---------------------------------------------------------------------------
# Cost projection (IMPLEMENTATION.md section 7)
# ---------------------------------------------------------------------------
def _profile(**per_condition: tuple[int, int]) -> TokenProfile:
    profile = TokenProfile()
    for name, (inp, out) in per_condition.items():
        profile.record(
            RunTokens(
                condition=Condition(name),
                scenario=Scenario.RESOURCE_SCARCITY,
                experiment=ExperimentRegime.EQUAL_ACTIVITY,
                seed=1,
                input_tokens=inp,
                output_tokens=out,
            )
        )
    return profile


def test_cost_is_projected_per_condition_not_pooled() -> None:
    """A and C cost more by design; an average would price no actual run."""
    config = campaign_config()
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    profile = _profile(A=(3_000_000, 100_000), B=(1_000_000, 100_000), C=(2_000_000, 100_000))

    estimate = estimate_cost(plan, profile, PRICING, model="gpt-5")

    seeds = len(config.seeds.pilot)
    assert estimate.input_tokens == (3_000_000 + 1_000_000 + 2_000_000) * seeds
    assert estimate.total_runs == seeds * 3


@pytest.mark.invariant
def test_pricing_an_unmeasured_condition_is_refused_not_guessed() -> None:
    """A fabricated component makes the total useless for authorising a batch."""
    config = campaign_config()
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    profile = _profile(A=(1_000, 100), B=(1_000, 100))  # no C

    with pytest.raises(CostError, match="condition C"):
        estimate_cost(plan, profile, PRICING, model="gpt-5")


def test_cells_priced_by_extrapolation_are_named() -> None:
    """An extrapolated cell must never look like a measured one."""
    config = campaign_config()
    plan = build_plan(config, phase="pilot")
    profile = _profile(A=(1_000, 100), B=(1_000, 100), C=(1_000, 100))

    estimate = estimate_cost(plan, profile, PRICING, model="gpt-5")

    # Three secondary scenarios x three conditions, none measured in the profile.
    assert len(estimate.extrapolated_cells) == 9
    assert any("priced by extrapolation" in w for w in estimate.warnings)


def test_the_upper_bound_only_inflates_output_tokens() -> None:
    """Input length is known exactly; only the model's reply length is not."""
    config = campaign_config()
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    profile = _profile(A=(1_000_000, 0), B=(1_000_000, 0), C=(1_000_000, 0))

    estimate = estimate_cost(plan, profile, PRICING, model="gpt-5")

    assert estimate.output_cost == 0
    assert estimate.upper_bound == estimate.total_cost


# ---------------------------------------------------------------------------
# The cost gate (IMPLEMENTATION.md section 7: never launch an unpriced real batch)
# ---------------------------------------------------------------------------
def _write_config(path: Path, *, mock: bool, seeds: list[int]) -> Path:
    """A tiny on-disk config, so the gate is exercised through the real CLI."""
    import yaml

    config = campaign_config(
        seeds={"pilot": seeds, "final": list(range(101, 116))},
        llm={"mock": mock},
    )
    path.write_text(yaml.safe_dump(json.loads(config.model_dump_json())), encoding="utf-8")
    return path


@pytest.mark.invariant
def test_a_real_provider_batch_without_an_estimate_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Spending on a real provider cannot be undone by re-running."""
    config = _write_config(tmp_path / "real.yaml", mock=False, seeds=PILOT_SEEDS)

    code = main(
        [
            "campaign-run",
            "--config",
            str(config),
            "--phase",
            "pilot",
            "--out",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 5
    assert "without a cost estimate" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.invariant
def test_a_real_provider_batch_needs_the_exact_projected_total(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Confirmation is by value, so it cannot carry over from a smaller plan.

    A boolean --yes would be satisfied by a habit; repeating the printed number
    requires having read the number.
    """
    mock_logs = tmp_path / "profile"
    mock_config = _write_config(tmp_path / "mock.yaml", mock=True, seeds=PILOT_SEEDS)
    assert (
        main(
            [
                "campaign-run",
                "--config",
                str(mock_config),
                "--phase",
                "pilot",
                "--out",
                str(mock_logs),
            ]
        )
        == 0
    )
    capsys.readouterr()

    config = _write_config(tmp_path / "real.yaml", mock=False, seeds=PILOT_SEEDS)
    argv = [
        "campaign-run",
        "--config",
        str(config),
        "--phase",
        "pilot",
        "--out",
        str(tmp_path / "runs"),
        "--profile-logs",
        str(mock_logs),
    ]

    # Without an acknowledged provider-side spend limit it is refused first.
    assert main([*argv, "--confirm-cost", "0.01"]) == 5
    assert "spend limit" in capsys.readouterr().err

    assert main([*argv, "--spend-limit-set", "--confirm-cost", "0.01"]) == 5
    error = capsys.readouterr().err
    assert "without confirmation" in error
    assert not (tmp_path / "runs").exists()


def test_a_mock_campaign_runs_without_confirmation(tmp_path: Path) -> None:
    """Cost discipline must not obstruct the offline path, which is the default."""
    config = campaign_config(seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))})
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))

    result = run_campaign(config, plan, tmp_path / "runs")

    assert len(result.completed) == len(PILOT_SEEDS)
    for trio in plan.trios:
        directory = trio_directory(tmp_path / "runs", trio)
        for condition in (Condition.A, Condition.B, Condition.C):
            assert (directory / f"{condition.value}.jsonl").exists()


def test_a_campaign_resumes_rather_than_re_running(tmp_path: Path) -> None:
    config = campaign_config(seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))})
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))

    run_campaign(config, plan, tmp_path / "runs")
    second = run_campaign(config, plan, tmp_path / "runs")

    assert second.completed == []
    assert len(second.skipped) == len(PILOT_SEEDS)


@pytest.mark.invariant
def test_every_cell_of_a_campaign_gets_its_own_directory(tmp_path: Path) -> None:
    """The same seed runs in several cells; a flat layout would overwrite logs."""
    config = campaign_config(
        seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))},
        budget={"experiment_2_total_tokens": 1_000_000},
    )
    plan = build_plan(config, phase="final")

    directories = {trio_directory(tmp_path, trio) for trio in plan.trios}
    assert len(directories) == len(plan.trios)


@pytest.mark.invariant
def test_a_campaign_run_carries_its_planned_scenario_and_experiment(tmp_path: Path) -> None:
    """The plan must actually reach the run, not merely describe it."""
    from rpg_economy.logging_ import read_metadata

    config = campaign_config(seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))})
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.DEMAND_SURGE,))
    run_campaign(config, plan, tmp_path / "runs")

    directory = trio_directory(tmp_path / "runs", plan.trios[0])
    metadata = read_metadata(directory / "A.jsonl")
    assert metadata.scenario is Scenario.DEMAND_SURGE
    assert metadata.seed == 1


def test_a_profile_is_measured_from_logs_not_live_state(tmp_path: Path) -> None:
    """So a batch can be priced from any past mock run without re-running it."""
    config = campaign_config(seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))})
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    run_campaign(config, plan, tmp_path / "runs")

    profile = measure_profile(tmp_path / "runs")

    assert profile.runs_measured == len(PILOT_SEEDS) * 3
    for condition in (Condition.A, Condition.B, Condition.C):
        measurement = profile.median_for(
            Scenario.RESOURCE_SCARCITY, ExperimentRegime.EQUAL_ACTIVITY, condition
        )
        assert measurement is not None
        assert measurement[0] > 0


def test_a_directory_of_unrelated_jsonl_is_skipped_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "notes.jsonl").write_text('{"hello": "world"}\n', encoding="utf-8")

    assert measure_profile(tmp_path).runs_measured == 0


# ---------------------------------------------------------------------------
# The de-risk comparison gates (design s13, s15)
# ---------------------------------------------------------------------------
def _real_provider_config(tmp_path: Path, seeds: list[int]) -> Path:
    return _write_config(tmp_path / "real.yaml", mock=False, seeds=seeds)


@pytest.mark.invariant
def test_the_derisk_run_refuses_without_an_acknowledged_spend_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """This repository can refuse an unpriced batch; it cannot cap the bill.

    Pricing proves the batch was understood. It does not prevent a runaway loop
    from spending more than the projection, which only a provider-side limit
    can, so the acknowledgement is a separate gate rather than folded into the
    cost confirmation.
    """
    mock_logs = tmp_path / "profile"
    mock_config = _write_config(tmp_path / "mock.yaml", mock=True, seeds=PILOT_SEEDS)
    assert (
        main(
            [
                "campaign-run",
                "--config",
                str(mock_config),
                "--phase",
                "pilot",
                "--out",
                str(mock_logs),
            ]
        )
        == 0
    )
    capsys.readouterr()

    config = _real_provider_config(tmp_path, PILOT_SEEDS)
    code = main(
        [
            "derisk",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "derisk"),
            "--profile-logs",
            str(mock_logs),
            "--model-pricing",
            "gpt-5-mini=0.25/2.00",
            "--confirm-cost",
            "999.99",
        ]
    )

    assert code == 5
    assert "spend limit" in capsys.readouterr().err
    assert not (tmp_path / "derisk").exists()


@pytest.mark.invariant
def test_the_derisk_run_refuses_without_a_cost_estimate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _real_provider_config(tmp_path, PILOT_SEEDS)

    code = main(
        [
            "derisk",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "derisk"),
            "--spend-limit-set",
        ]
    )

    assert code == 5
    assert "without a cost estimate" in capsys.readouterr().err
    assert not (tmp_path / "derisk").exists()


def test_derisk_rates_are_computed_from_logs(tmp_path: Path) -> None:
    """The three rates come off the log, so a comparison can be re-reported."""
    from rpg_economy.campaign.derisk import collect_comparison, rates_from_log

    config = campaign_config(seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))})
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    run_campaign(config, plan, tmp_path / "runs")

    directory = trio_directory(tmp_path / "runs", plan.trios[0])
    rates = rates_from_log(directory / "A.jsonl", "mock")

    assert rates.condition == "A"
    assert rates.llm_calls > 0
    assert rates.director_calls > 0
    assert 0.0 <= rates.invalid_output_rate <= 1.0
    assert 0.0 <= rates.wasted_activation_rate <= 1.0

    # Laid out as the de-risk command expects, so it can be re-read for free.
    layout = tmp_path / "layout"
    (layout / "mock").mkdir(parents=True)
    for condition in ("A", "B", "C"):
        (layout / "mock" / f"{condition}.jsonl").write_bytes(
            (directory / f"{condition}.jsonl").read_bytes()
        )
    comparison = collect_comparison(layout, ("mock",), seed=1, horizon=12)
    assert len(comparison.rates) == 3


def test_the_derisk_config_keeps_all_three_phases(tmp_path: Path) -> None:
    """A comparison that skipped the shock would not exercise the protocol."""
    from rpg_economy.campaign.derisk import derisk_config

    config = derisk_config(campaign_config(), model="gpt-5-nano", horizon=120)

    assert config.time.horizon_ticks == 120
    assert config.phases.base_ticks > 0
    assert config.phases.shock_ticks > 0
    assert config.phases.recovery_ticks > 0
    assert config.llm.model == "gpt-5-nano"
    assert config.llm.mock is False


# ---------------------------------------------------------------------------
# Parallel campaign execution (IMPLEMENTATION.md section 5)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_parallel_execution_does_not_change_any_run(tmp_path: Path) -> None:
    """The whole point: faster, and byte-identical.

    A campaign that ran quicker and produced different numbers would be
    worthless, so this compares the logs a parallel run produces against the
    serial ones for the same plan.
    """
    from rpg_economy.campaign.parallel import run_campaign_parallel
    from rpg_economy.logging_ import log_digest

    config = campaign_config(seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))})
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))

    run_campaign(config, plan, tmp_path / "serial")
    run_campaign_parallel(config, plan, tmp_path / "parallel", workers=4)

    for trio in plan.trios:
        for condition in (Condition.A, Condition.B, Condition.C):
            serial = trio_directory(tmp_path / "serial", trio) / f"{condition.value}.jsonl"
            par = trio_directory(tmp_path / "parallel", trio) / f"{condition.value}.jsonl"
            assert log_digest(serial) == log_digest(par), (
                f"{trio.seed}/{condition.value} differs between serial and parallel"
            )


@pytest.mark.invariant
def test_a_resumed_campaign_never_reruns_a_finished_trio(tmp_path: Path) -> None:
    """With a real provider a restart from zero is not an inconvenience, it is money."""
    from rpg_economy.campaign.parallel import run_campaign_parallel

    config = campaign_config(seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))})
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))

    first = run_campaign_parallel(config, plan, tmp_path / "runs", workers=4)
    assert len(first.by_status("completed")) == len(PILOT_SEEDS)

    second = run_campaign_parallel(config, plan, tmp_path / "runs", workers=4)
    assert second.by_status("completed") == []
    assert len(second.by_status("skipped")) == len(PILOT_SEEDS)


def test_only_transient_provider_failures_are_retried() -> None:
    """Rate limits and outages are waited out; refusals and bugs are not."""
    from rpg_economy.campaign.parallel import is_retryable
    from rpg_economy.llm import ProviderError

    assert is_retryable(ProviderError("still failing after 5 retries: 503", transient=True))
    assert not is_retryable(ProviderError("Error code: 401 - invalid_api_key"))
    assert not is_retryable(ProviderError("insufficient_quota"))
    # A message that merely mentions a status code is not a provider failure.
    assert not is_retryable(RuntimeError("Error code: 429 - rate limit exceeded"))
    assert not is_retryable(ValueError("condition must be a Condition enum member"))


def test_backoff_is_jittered_and_capped() -> None:
    """Eight workers backing off for exactly the same interval return in lockstep
    and re-trigger the same rate limit together."""
    import random as _random

    from rpg_economy.campaign.parallel import BACKOFF_CAP_SECONDS, backoff_delay

    draws = [backoff_delay(3, rng=_random.Random(seed)) for seed in range(20)]
    assert len(set(draws)) > 1, "backoff is not jittered"
    assert all(0 <= d <= BACKOFF_CAP_SECONDS for d in draws)
    assert backoff_delay(99, rng=_random.Random(0)) <= BACKOFF_CAP_SECONDS
