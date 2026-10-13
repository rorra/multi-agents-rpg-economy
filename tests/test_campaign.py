"""Campaign planning, pricing and execution (IMPLEMENTATION.md phase 7)."""

from __future__ import annotations

import dataclasses
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


@pytest.mark.invariant
def test_a_truncated_log_is_not_treated_as_finished(tmp_path: Path) -> None:
    """Only a log that reached run_end is finished; a cut-off one never is.

    Asserted on the log checks themselves: the trio's integrity marker would
    reject any edited log on its own, and so would prove nothing about them.
    """
    from rpg_economy.campaign.runner import condition_log_matches
    from rpg_economy.logging_ import log_is_finished

    config = campaign_config()
    plan, trio, cell = _one_trio(config)
    run_campaign(config, plan, tmp_path / "runs")
    directory = trio_directory(tmp_path / "runs", trio)
    log = directory / "A.jsonl"
    assert log_is_finished(log)
    assert condition_log_matches(directory, Condition.A, cell)

    whole = log.read_bytes()
    lines = whole.splitlines(keepends=True)
    cases = {
        "a log without its run_end is not finished": b"".join(lines[:-1]),
        "a log cut mid-line is not finished": whole[:-15],
        "a non-empty file is not a finished run": b"{}\n",
    }
    for reason, content in cases.items():
        log.write_bytes(content)
        assert not log_is_finished(log), reason
        assert not condition_log_matches(directory, Condition.A), reason
        assert not condition_log_matches(directory, Condition.A, cell), reason


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


@pytest.mark.invariant
def test_a_parallel_trio_is_identical_to_a_serial_one(tmp_path: Path) -> None:
    """Condition-level parallelism must not change a run either.

    A trio's critical path is otherwise A + B + C in series and A is the longest,
    so this is where most of the wall-clock saving lives -- which makes it the
    place a difference would matter most.
    """
    from rpg_economy.campaign.parallel import run_trio_parallel
    from rpg_economy.logging_ import log_digest
    from rpg_economy.runner import run_trio

    config = campaign_config()
    run_trio(config, tmp_path / "serial", seed=1004)
    run_trio_parallel(config, tmp_path / "parallel", seed=1004)

    for condition in (Condition.A, Condition.B, Condition.C):
        serial = tmp_path / "serial" / f"{condition.value}.jsonl"
        par = tmp_path / "parallel" / f"{condition.value}.jsonl"
        assert log_digest(serial) == log_digest(par), f"{condition.value} differs"


@pytest.mark.invariant
def test_a_parallel_trio_still_passes_the_integrity_gate(tmp_path: Path) -> None:
    """The trio integrity gate (design s12) applies however the trio was scheduled."""
    from rpg_economy.campaign.parallel import run_trio_parallel

    runs = run_trio_parallel(campaign_config(), tmp_path / "runs", seed=1004)
    assert set(runs) == {Condition.A, Condition.B, Condition.C}


# ---------------------------------------------------------------------------
# Billing provenance (IMPLEMENTATION.md section 4.7)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_mock_call_can_never_enter_the_billed_ledger(tmp_path: Path) -> None:
    """Real and mock spend must be structurally unmixable.

    Configuration cannot be the guard -- a config can say ``mock: false`` while
    the run executes offline -- so the flag comes from the provider response,
    and a mock response can never set it.
    """
    config = campaign_config(run={"condition": "A"})
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    run_campaign(config, plan, tmp_path / "runs")

    directory = trio_directory(tmp_path / "runs", plan.trios[0])
    for condition in (Condition.A, Condition.B, Condition.C):
        events = [
            json.loads(line)
            for line in (directory / f"{condition.value}.jsonl").read_text().splitlines()
        ]
        consumed = next(e for e in events if e["type"] == "tokens_consumed")
        assert consumed["payload"]["authoritative_total"] > 0, "the run made no calls"
        assert consumed["payload"]["billed_total"] == 0, "mock spend entered the billed ledger"
        assert consumed["payload"]["billed_calls"] == 0


@pytest.mark.invariant
def test_billing_provenance_comes_from_the_response_not_the_config() -> None:
    """``mock: false`` is a statement of intent, not evidence of a billed call."""
    from rpg_economy.llm.client import LLMResponse, MockLLMClient

    mock = MockLLMClient(lambda system, user: '{"action":"wait","short_reason":"x"}')
    assert mock.complete(system="s", user="u").billed is False

    # The default is unbilled, so any client that forgets to set it is safe.
    assert LLMResponse(text="", input_tokens=1, output_tokens=1, model="m").billed is False


@pytest.mark.invariant
def test_an_unbilled_call_is_absent_from_billed_totals_but_not_from_the_ledger() -> None:
    """The Experiment 2 budget still binds on every call; only *spend* is billed."""
    from rpg_economy.domain import TokenBucket
    from rpg_economy.llm.tokens import TokenLedger

    ledger = TokenLedger()
    ledger.record(
        input_by_bucket={TokenBucket.AGENT: 100},
        output_tokens=10,
        output_bucket=TokenBucket.AGENT,
        authoritative_total=110,
        billed=False,
    )
    ledger.record(
        input_by_bucket={TokenBucket.AGENT: 200},
        output_tokens=20,
        output_bucket=TokenBucket.AGENT,
        authoritative_total=220,
        billed=True,
    )

    assert ledger.authoritative_total == 330, "the budget must see every call"
    assert ledger.billed_total == 220, "only the billed call is spend"
    assert ledger.billed_calls == 1


# ---------------------------------------------------------------------------
# The two-stage pilot (design s11.4, s15)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_budget_derivation_runs_at_the_full_horizon() -> None:
    """Design v1.0 s8.4 pools runs at the horizon the finals will use.

    A budget derived from compressed runs would be a fraction of what a final run
    needs, and every Experiment 2 run would exhaust almost immediately.
    """
    from rpg_economy.campaign.pilot import BUDGET_DERIVATION, CALIBRATION, stage_config

    base = campaign_config()
    assert stage_config(base, BUDGET_DERIVATION).time.horizon_ticks == base.time.horizon_ticks
    assert BUDGET_DERIVATION.uses_full_horizon
    # Calibration may compress, because its output is a judgement not a number.
    assert not CALIBRATION.uses_full_horizon


@pytest.mark.invariant
def test_a_stage_derives_only_the_horizon_from_the_base_config() -> None:
    """One recorded dimension of difference, and no other.

    Everything that governs behaviour -- K, population, recipes, thresholds,
    traits, prompts, model, inference parameters -- must come from the base
    configuration untouched, or the pilot is not testing it.
    """
    from rpg_economy.campaign.pilot import CALIBRATION, stage_config

    base = campaign_config()
    derived = stage_config(base, CALIBRATION)

    before = base.model_dump(mode="python")
    after = derived.model_dump(mode="python")
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {"time", "phases"}, f"a stage changed more than the horizon: {changed}"
    assert after["time"]["k_activations"] == before["time"]["k_activations"]
    assert after["llm"] == before["llm"]
    assert after["agents"] == before["agents"]
    assert after["economy"] == before["economy"]


@pytest.mark.invariant
def test_a_compressed_stage_keeps_every_phase_measurable() -> None:
    """A horizon that cannot express recovery would censor every observation."""
    from rpg_economy.campaign.pilot import CALIBRATION, PilotStage, stage_config

    base = campaign_config()
    derived = stage_config(base, CALIBRATION)

    assert derived.phases.base_ticks >= derived.metrics.volatility_window_w
    assert derived.phases.recovery_ticks >= derived.metrics.recovery_hold_ticks_m
    assert derived.phases.shock_ticks > 0

    # Derived from the fixture rather than hardcoded: a literal horizon that is
    # "obviously too short" depends on W and M, which differ between the test
    # config and the base one.
    too_short = max(3, base.metrics.volatility_window_w)
    impossible = PilotStage(
        name="too-short", horizon=too_short, scenarios=None, purpose="x", purpose_key="x"
    )
    with pytest.raises(ValueError, match=r"volatility window|hold requirement"):
        stage_config(base, impossible)


def test_stage_billing_verification_reads_logs(tmp_path: Path) -> None:
    """Billing provenance applied to a whole stage: every run real and fully billed."""
    from rpg_economy.campaign.pilot import verify_stage_billing

    config = campaign_config(seeds={"pilot": PILOT_SEEDS, "final": list(range(101, 116))})
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    run_campaign(config, plan, tmp_path / "runs")

    report = verify_stage_billing(tmp_path / "runs")

    assert report["runs"] == len(PILOT_SEEDS) * 3
    # Mock runs: real in no sense, billed in no sense. Both flags must be False.
    assert report["all_real"] is False
    assert report["all_fully_billed"] is False
    assert report["billed_total"] == 0


@pytest.mark.invariant
def test_an_unfinished_trio_is_set_aside_and_run_again(tmp_path: Path) -> None:
    """Resuming re-runs an unfinished trio and keeps its partial logs, never deletes them."""
    config = campaign_config()
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    plan = dataclasses.replace(plan, trios=plan.trios[:1])
    run_campaign(config, plan, tmp_path / "runs")

    directory = trio_directory(tmp_path / "runs", plan.trios[0])
    log = directory / "B.jsonl"
    cut = b"".join(log.read_bytes().splitlines(keepends=True)[:-3])
    log.write_bytes(cut)

    second = run_campaign(config, plan, tmp_path / "runs")
    assert second.completed == plan.trios
    assert (directory / "previous-1" / "B.jsonl").read_bytes() == cut
    from rpg_economy.campaign.parallel import trio_is_complete

    assert trio_is_complete(directory)


def test_an_unfinished_log_is_left_out_of_a_profile(tmp_path: Path) -> None:
    config = campaign_config()
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    plan = dataclasses.replace(plan, trios=plan.trios[:1])
    run_campaign(config, plan, tmp_path / "runs")
    log = trio_directory(tmp_path / "runs", plan.trios[0]) / "A.jsonl"
    log.write_bytes(b"".join(log.read_bytes().splitlines(keepends=True)[:-2]))

    profile = measure_profile(tmp_path / "runs")
    assert profile.runs_measured == 2
    assert profile.skipped == 1


@pytest.mark.invariant
def test_the_estimate_is_priced_at_the_planned_horizon(tmp_path: Path) -> None:
    """A profile of short runs must not price a long campaign as if it were short."""
    config = campaign_config()
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    run_campaign(config, plan, tmp_path / "runs")
    profile = measure_profile(tmp_path / "runs")

    short = estimate_cost(plan, profile, PRICING, model="gpt-5", horizon=12)
    double = estimate_cost(plan, profile, PRICING, model="gpt-5", horizon=24)
    assert abs(double.input_tokens - 2 * short.input_tokens) <= plan.total_runs
    assert double.total_cost > short.total_cost


# ---------------------------------------------------------------------------
# A finished trio counts only for the setup it ran
# ---------------------------------------------------------------------------
def _one_trio(config):
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    plan = dataclasses.replace(plan, trios=plan.trios[:1])
    trio = plan.trios[0]
    cell = config.for_cell(scenario=trio.scenario, experiment=trio.experiment, seed=trio.seed)
    return plan, trio, cell


@pytest.mark.invariant
def test_a_finished_trio_is_complete_only_for_the_setup_it_ran(tmp_path: Path) -> None:
    from rpg_economy.campaign.runner import trio_is_complete

    config = campaign_config()
    plan, trio, cell = _one_trio(config)
    run_campaign(config, plan, tmp_path / "runs")
    directory = trio_directory(tmp_path / "runs", trio)

    assert trio_is_complete(directory)
    assert trio_is_complete(directory, cell)

    longer = campaign_config(
        phases={"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 8},
        time={"horizon_ticks": 16},
    )
    _, _, longer_cell = _one_trio(longer)
    assert not trio_is_complete(directory, longer_cell), "another setup's trio was accepted"


@pytest.mark.invariant
def test_a_mock_trio_never_stands_in_for_a_real_one(tmp_path: Path) -> None:
    """The hash ignores the mode, so the mode is checked on its own."""
    from rpg_economy.campaign.runner import trio_is_complete
    from rpg_economy.cli import with_real_provider
    from rpg_economy.config import config_hash

    config = campaign_config()
    plan, trio, cell = _one_trio(config)
    run_campaign(config, plan, tmp_path / "runs")
    directory = trio_directory(tmp_path / "runs", trio)

    real = with_real_provider(cell)
    assert config_hash(real) == config_hash(cell)
    assert not trio_is_complete(directory, real)


def test_a_resumed_campaign_reruns_a_trio_from_another_setup(tmp_path: Path) -> None:
    config = campaign_config()
    plan, _, _ = _one_trio(config)
    run_campaign(config, plan, tmp_path / "runs")

    longer = campaign_config(
        phases={"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 8},
        time={"horizon_ticks": 16},
    )
    longer_plan, _, _ = _one_trio(longer)
    second = run_campaign(longer, longer_plan, tmp_path / "runs")

    assert second.skipped == []
    assert second.completed == longer_plan.trios


# ---------------------------------------------------------------------------
# The worst case and the Experiment 2 budget in the estimate
# ---------------------------------------------------------------------------
def test_the_worst_case_covers_the_projected_cost() -> None:
    config = campaign_config(budget={"experiment_2_total_tokens": 50_000})
    plan = build_plan(config, phase="final", scenarios=(Scenario.RESOURCE_SCARCITY,))
    profile = _profile(A=(40_000, 4_000), B=(20_000, 2_000), C=(30_000, 3_000))

    estimate = estimate_cost(plan, profile, PRICING, model="gpt-5", config=config)

    assert estimate.worst_case is not None
    assert estimate.worst_case >= estimate.total_cost
    assert estimate.to_payload()["worst_case"] == str(estimate.worst_case)


def test_without_a_config_there_is_no_worst_case() -> None:
    config = campaign_config()
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    profile = _profile(A=(1_000, 100), B=(1_000, 100), C=(1_000, 100))

    estimate = estimate_cost(plan, profile, PRICING, model="gpt-5")

    assert estimate.worst_case is None
    assert estimate.to_payload()["worst_case"] is None


@pytest.mark.invariant
def test_an_experiment_2_run_is_priced_no_higher_than_its_budget() -> None:
    """A run stops calling once its token budget is spent."""
    cap = 50_000
    config = campaign_config(budget={"experiment_2_total_tokens": cap})
    plan = build_plan(config, phase="final", scenarios=(Scenario.RESOURCE_SCARCITY,))
    profile = _profile(A=(1_000_000, 100_000), B=(1_000_000, 100_000), C=(1_000_000, 100_000))

    capped = estimate_cost(plan, profile, PRICING, model="gpt-5", config=config)
    uncapped = estimate_cost(plan, profile, PRICING, model="gpt-5")

    exp1_runs = sum(
        len(t.conditions) for t in plan.trios if t.experiment is ExperimentRegime.EQUAL_ACTIVITY
    )
    exp2_runs = sum(
        len(t.conditions) for t in plan.trios if t.experiment is ExperimentRegime.EQUAL_COST
    )
    assert exp2_runs > 0
    bucket = capped.per_scenario[Scenario.RESOURCE_SCARCITY.value]
    assert bucket["input_tokens"] + bucket["output_tokens"] == exp1_runs * 1_100_000 + (
        exp2_runs * cap
    )
    assert capped.output_tokens < uncapped.output_tokens
    assert capped.total_cost < uncapped.total_cost
    # The headline totals are the per-scenario buckets summed, capped runs included.
    assert capped.input_tokens == sum(b["input_tokens"] for b in capped.per_scenario.values())
    assert capped.output_tokens == sum(b["output_tokens"] for b in capped.per_scenario.values())


# ---------------------------------------------------------------------------
# Resume never pays again for a finished trio
# ---------------------------------------------------------------------------
def _single_trio_plan(config, seed_index: int = 0):
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    return dataclasses.replace(plan, trios=[plan.trios[seed_index]])


@pytest.mark.invariant
def test_pricing_and_transport_are_not_part_of_the_config_hash() -> None:
    """A price or timeout edited between a crash and its resume changes no run."""
    from rpg_economy.config import config_hash, config_payload

    config = campaign_config()
    edited = campaign_config(
        llm={
            "pricing": {
                "input_per_1m_tokens": "9.99",
                "output_per_1m_tokens": "99.99",
                "currency": "USD",
                "quoted_on": "2027-01-01",
            },
            "request_timeout_s": 5.0,
            "max_transient_retries": 1,
            "retry_base_delay_s": 0.5,
            "retry_max_delay_s": 1.0,
        }
    )

    assert config_hash(edited) == config_hash(config)
    for name in ("pricing", "request_timeout_s", "max_transient_retries", "mock"):
        assert name not in config_payload(config)["llm"]
    # What the model sees still counts.
    assert config_hash(campaign_config(llm={"max_output_tokens": 77})) != config_hash(config)


@pytest.mark.invariant
def test_a_price_update_does_not_rerun_a_finished_trio(tmp_path: Path) -> None:
    config = campaign_config()
    plan = _single_trio_plan(config)
    run_campaign(config, plan, tmp_path / "runs")

    repriced = campaign_config(
        llm={
            "pricing": {
                "input_per_1m_tokens": "2.00",
                "output_per_1m_tokens": "20.00",
                "quoted_on": "2027-01-01",
            },
            "request_timeout_s": 10.0,
        }
    )
    second = run_campaign(repriced, plan, tmp_path / "runs")
    assert second.completed == []
    assert second.skipped == plan.trios


@pytest.mark.invariant
def test_a_trio_is_complete_only_for_its_own_seed(tmp_path: Path) -> None:
    """The cell's seed is the reference, not whatever seed the log reports."""
    import shutil

    from rpg_economy.campaign.runner import trio_is_complete

    config = campaign_config()
    plan = _single_trio_plan(config)
    run_campaign(config, plan, tmp_path / "runs")
    trio = plan.trios[0]
    finished = trio_directory(tmp_path / "runs", trio)

    other = dataclasses.replace(trio, seed=trio.seed + 1)
    misplaced = trio_directory(tmp_path / "runs", other)
    shutil.copytree(finished, misplaced)
    cell = config.for_cell(scenario=other.scenario, experiment=other.experiment, seed=other.seed)
    assert not trio_is_complete(misplaced, cell), "a trio of another seed was accepted"


def test_a_resumed_batch_is_priced_only_on_the_trios_still_to_run(tmp_path: Path) -> None:
    from rpg_economy.campaign.runner import pending_trios

    config = campaign_config()
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    run_campaign(config, dataclasses.replace(plan, trios=plan.trios[:3]), tmp_path / "runs")

    pending = pending_trios(config, plan, tmp_path / "runs", resume=True)
    assert pending.trios == plan.trios[3:]
    assert pending_trios(config, plan, tmp_path / "runs", resume=False).trios == plan.trios

    profile = _profile(A=(1_000, 100), B=(1_000, 100), C=(1_000, 100))
    full = estimate_cost(plan, profile, PRICING, model="gpt-5", config=config)
    remaining = estimate_cost(pending, profile, PRICING, model="gpt-5", config=config)
    assert remaining.total_runs == 3 * len(pending.trios)
    assert remaining.total_cost < full.total_cost


def test_a_resumed_campaign_run_says_what_it_prices(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_config(tmp_path / "mock.yaml", mock=True, seeds=PILOT_SEEDS)
    argv = ["campaign-run", "--config", str(config), "--phase", "pilot"]
    argv += ["--out", str(tmp_path / "runs")]
    assert main(argv) == 0
    capsys.readouterr()

    assert main([*argv, "--profile-logs", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "already complete and verified" in out


# ---------------------------------------------------------------------------
# The integrity marker: a trio that failed the gate is never skipped
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_verified_trio_carries_a_marker_pinned_to_its_logs(tmp_path: Path) -> None:
    from rpg_economy.campaign.runner import trio_is_complete
    from rpg_economy.integrity import INTEGRITY_MARKER, has_integrity_marker

    config = campaign_config()
    plan = _single_trio_plan(config)
    run_campaign(config, plan, tmp_path / "runs")
    directory = trio_directory(tmp_path / "runs", plan.trios[0])

    assert (directory / INTEGRITY_MARKER).exists()
    assert has_integrity_marker(directory)

    # A log changed after the check no longer carries the verdict.
    log = directory / "B.jsonl"
    original = log.read_bytes()
    log.write_bytes(original + b"\n")
    assert not has_integrity_marker(directory)
    assert not trio_is_complete(directory)
    log.write_bytes(original)
    assert trio_is_complete(directory)

    (directory / INTEGRITY_MARKER).unlink()
    assert not trio_is_complete(directory), "a trio without a verdict was complete"


@pytest.mark.invariant
def test_a_trio_that_failed_the_gate_is_not_skipped_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rpg_economy import integrity
    from rpg_economy.campaign.parallel import run_campaign_parallel
    from rpg_economy.campaign.runner import trio_is_complete

    config = campaign_config()
    plan = _single_trio_plan(config)
    directory = trio_directory(tmp_path / "runs", plan.trios[0])
    real_verify = integrity.verify_trio

    def failing(runs):
        raise integrity.TrioIntegrityError("injected")

    monkeypatch.setattr(integrity, "verify_trio", failing)
    first = run_campaign_parallel(config, plan, tmp_path / "runs", workers=1)
    assert len(first.failed) == 1
    assert "injected" in first.failed[0].detail
    # Three finished logs, but no verdict: not complete.
    assert all((directory / f"{c.value}.jsonl").exists() for c in Condition)
    assert not trio_is_complete(directory)

    # Resumed after the defect is fixed: the gate runs again on the logs that
    # exist, and nothing is re-run to get there.
    monkeypatch.setattr(integrity, "verify_trio", real_verify)
    second = run_campaign_parallel(config, plan, tmp_path / "runs", workers=1)
    assert second.by_status("skipped") == []
    assert len(second.by_status("completed")) == 1
    assert trio_is_complete(directory)
    assert not list(directory.glob("previous-*"))
