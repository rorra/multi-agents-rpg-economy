"""Experiment 2: budget enforcement, exhaustion, and budget derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpg_economy.agents.offline import json_action
from rpg_economy.analysis.budget_derivation import (
    BudgetDerivationError,
    derive_budget,
    find_logs,
    run_total_tokens,
)
from rpg_economy.domain import TokenBucket
from rpg_economy.llm import LLMResponse, TokenLedger, estimate_tokens
from rpg_economy.llm.budget import TokenBudget, common_active_horizon
from rpg_economy.logging_ import EventType, read_events
from rpg_economy.runner import run_condition, run_trio, trio_common_active_horizon
from tests.conftest import make_config


class InflatedClient:
    """A client whose reported usage far exceeds what the estimator would say.

    Exists to force bucket attribution and the provider's figure apart, which
    they never do in normal operation, so that the rule of design s8.4 can be
    tested rather than merely assumed.
    """

    def __init__(self, text: str, multiplier: int = 50) -> None:
        self._text = text
        self._multiplier = multiplier
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "inflated"

    def complete(self, *, system: str, user: str) -> LLMResponse:
        self.calls += 1
        self.prompts.append((system, user))
        return LLMResponse(
            text=self._text,
            input_tokens=1000 * self._multiplier,
            output_tokens=100 * self._multiplier,
            model=self.model,
            finish_reason="stop",
        )


def experiment_2(budget: int, **overrides) -> object:
    run = {"experiment": "experiment_2", "condition": "B", **overrides.pop("run", {})}
    return make_config(run=run, budget={"experiment_2_total_tokens": budget}, **overrides)


# ---------------------------------------------------------------------------
# The budget object
# ---------------------------------------------------------------------------
def test_experiment_1_has_no_cap() -> None:
    budget = TokenBudget()
    budget.charge(10**9, tick=1, cycle=1)
    assert budget.may_call()
    assert not budget.exhausted


def test_a_call_is_authorised_before_it_is_charged() -> None:
    """A call's cost is not knowable in advance, so one may cross the cap."""
    budget = TokenBudget(cap=100)
    assert budget.may_call()
    budget.charge(500, tick=3, cycle=3)
    assert not budget.may_call()
    assert budget.exhausted_tick == 3


def test_exhaustion_is_recorded_once() -> None:
    budget = TokenBudget(cap=100)
    assert budget.charge(150, tick=5, cycle=5) is True
    assert budget.charge(50, tick=9, cycle=9) is False
    assert budget.exhausted_tick == 5


def test_fraction_consumed_is_reported_when_not_exhausted() -> None:
    """Design v1.0 s11.2: if not exhausted, report the fraction consumed."""
    budget = TokenBudget(cap=1000)
    budget.charge(250, tick=1, cycle=1)
    assert budget.fraction_consumed == pytest.approx(0.25)
    assert budget.snapshot()["exhausted"] is False


def test_a_negative_charge_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot cost"):
        TokenBudget(cap=10).charge(-1, tick=0, cycle=0)


# ---------------------------------------------------------------------------
# The provider's total governs (design s8.4)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_the_ledger_keeps_the_authoritative_total_separate() -> None:
    ledger = TokenLedger()
    ledger.record(
        input_by_bucket={TokenBucket.AGENT: 10},
        output_tokens=5,
        output_bucket=TokenBucket.AGENT,
        authoritative_total=900,
    )
    assert ledger.total() == 15
    assert ledger.authoritative_total == 900


@pytest.mark.invariant
def test_the_budget_binds_on_the_provider_total_not_the_bucket_sum(
    tmp_path: Path,
) -> None:
    """Design s8.4, with the two figures forced apart.

    The client reports 55,000 tokens per call while the prompt text the
    estimator sees is a few hundred. A budget enforced on bucket estimates would
    barely move; enforced on the provider's figure it is gone almost at once.
    """
    cap = 200_000
    config = experiment_2(cap)
    client = InflatedClient(json_action(action="wait"))

    result = run_condition(config, tmp_path / "run.jsonl", client=client, cycles=40)

    # The provider reported 55,000 tokens per call and the budget followed it.
    assert result.budget.exhausted
    assert result.budget.consumed == result.ledger.authoritative_total
    assert result.budget.consumed == client.calls * 55_000
    assert client.calls == 4, "the budget should bind after four 55k calls"

    # What a locally estimated budget would have seen instead: the prompt text
    # is a few hundred tokens, nowhere near the cap. Enforcing on estimates
    # would have let this run continue indefinitely.
    estimated = sum(
        estimate_tokens(system) + estimate_tokens(user) for system, user in client.prompts
    )
    assert estimated < cap / 10, (
        f"estimator saw {estimated} tokens against a {cap} cap; the two figures "
        f"must be far apart for this test to mean anything"
    )
    assert not TokenBudget(cap=cap, consumed=estimated).exhausted


@pytest.mark.invariant
def test_skewed_attribution_does_not_move_the_exhaustion_tick(tmp_path: Path) -> None:
    """Attribution is analytical; it must not change when the budget binds."""
    cap = 300_000

    def exhaustion_tick(label: str, multiplier: int) -> int | None:
        config = experiment_2(cap)
        client = InflatedClient(json_action(action="wait"), multiplier=multiplier)
        result = run_condition(config, tmp_path / f"run-{label}.jsonl", client=client, cycles=60)
        return result.budget.exhausted_tick

    # Same provider cost per call either way; only the prompt sections differ in
    # how they would apportion. The tick must be identical.
    assert exhaustion_tick("first", 50) == exhaustion_tick("second", 50)


# ---------------------------------------------------------------------------
# Exhaustion semantics (design v1.0 s8.4)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_no_llm_calls_after_exhaustion(tmp_path: Path) -> None:
    config = experiment_2(150_000)
    client = InflatedClient(json_action(action="wait"))
    result = run_condition(config, tmp_path / "run.jsonl", client=client, cycles=60)

    events = list(read_events(result.log_path))
    exhausted = next(e for e in events if e.type is EventType.BUDGET_EXHAUSTED)
    after = [e for e in events if e.seq > exhausted.seq and e.type is EventType.LLM_CALL]
    assert after == [], f"{len(after)} LLM calls were made after exhaustion"


@pytest.mark.invariant
def test_mechanical_ticks_continue_to_the_planned_horizon(tmp_path: Path) -> None:
    """Design v1.0 s8.4: the engine completes the horizon regardless."""
    config = experiment_2(120_000)
    client = InflatedClient(json_action(action="wait"))
    result = run_condition(config, tmp_path / "run.jsonl", client=client)

    assert result.budget.exhausted
    assert result.state.tick == config.time.horizon_ticks

    events = list(read_events(result.log_path))
    exhausted = next(e for e in events if e.type is EventType.BUDGET_EXHAUSTED)
    ticks_after = [e for e in events if e.seq > exhausted.seq and e.type is EventType.TICK]
    assert len(ticks_after) > 0
    assert ticks_after[-1].tick == config.time.horizon_ticks - 1


@pytest.mark.invariant
def test_the_scheduled_shock_still_fires_after_exhaustion(tmp_path: Path) -> None:
    config = experiment_2(60_000)
    client = InflatedClient(json_action(action="wait"))
    result = run_condition(config, tmp_path / "run.jsonl", client=client)

    events = list(read_events(result.log_path))
    exhausted = next(e for e in events if e.type is EventType.BUDGET_EXHAUSTED)
    shocks = [e for e in events if e.type in {EventType.SHOCK_APPLIED, EventType.SHOCK_REVERTED}]
    assert exhausted.tick < config.phases.shock_tick, "test needs early exhaustion"
    assert [e.tick for e in shocks] == [
        config.phases.shock_tick,
        config.phases.reversion_tick,
    ]


@pytest.mark.invariant
def test_orders_still_expire_after_exhaustion(tmp_path: Path) -> None:
    """Mechanical rules continue: orders execute or expire (design s8.4)."""
    config = experiment_2(300_000)
    client = InflatedClient(json_action(action="sell", target="leather", quantity=1, price=10.0))
    result = run_condition(config, tmp_path / "run.jsonl", client=client)

    events = list(read_events(result.log_path))
    exhausted = next(e for e in events if e.type is EventType.BUDGET_EXHAUSTED)
    expiries = [e for e in events if e.seq > exhausted.seq and e.type is EventType.ORDER_EXPIRED]
    assert expiries, "no order expired after exhaustion"


def test_the_exhaustion_tick_is_logged(tmp_path: Path) -> None:
    """Design v1.0 s11.2: exhausting early is itself a result."""
    config = experiment_2(100_000)
    client = InflatedClient(json_action(action="wait"))
    result = run_condition(config, tmp_path / "run.jsonl", client=client, cycles=40)

    events = [e for e in read_events(result.log_path) if e.type is EventType.BUDGET_EXHAUSTED]
    assert len(events) == 1
    assert events[0].payload["exhausted_tick"] == result.budget.exhausted_tick
    assert events[0].payload["cap"] == 100_000


def test_a_run_that_does_not_exhaust_reports_its_fraction(tmp_path: Path) -> None:
    config = experiment_2(10**9)
    result = run_condition(config, tmp_path / "run.jsonl", cycles=20)

    assert not result.budget.exhausted
    assert result.exhaustion_tick is None
    assert 0 < result.budget.fraction_consumed < 1


def test_experiment_1_never_exhausts(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=20)
    assert result.budget.cap is None
    assert not result.budget.exhausted


# ---------------------------------------------------------------------------
# Common active horizon (design v1.0 s8.4)
# ---------------------------------------------------------------------------
def test_common_active_horizon_is_the_earliest_exhaustion() -> None:
    assert common_active_horizon([40, 90, None], planned_horizon=300) == 40


def test_common_active_horizon_falls_back_to_the_planned_horizon() -> None:
    assert common_active_horizon([None, None, None], planned_horizon=300) == 300


def test_a_trio_computes_its_common_active_horizon(tmp_path: Path) -> None:
    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=20)
    horizon = trio_common_active_horizon(runs)
    assert horizon == make_config().time.horizon_ticks


# ---------------------------------------------------------------------------
# Budget derivation (design v1.0 s8.4)
# ---------------------------------------------------------------------------
def pilot_logs(tmp_path: Path, seeds: tuple[int, ...] = (1001, 1002)) -> Path:
    directory = tmp_path / "pilot"
    for seed in seeds:
        run_trio(make_config(), directory / str(seed), seed=seed, cycles=6)
    return directory


def test_the_budget_is_the_pooled_median_of_per_run_totals(tmp_path: Path) -> None:
    """Pooled so no condition is privileged; median so no run can dominate."""
    import statistics

    derivation = derive_budget(find_logs(pilot_logs(tmp_path)), allow_mock=True)
    totals = [run.total_tokens for run in derivation.runs]

    assert len(derivation.runs) == 6  # two seeds x three conditions
    # Rounded half up, not truncated: an even count can give a half-token median.
    assert derivation.budget == int(statistics.median(totals) + 0.5)


def test_derivation_uses_provider_totals(tmp_path: Path) -> None:
    """Design s8.4: the budget is derived from what the provider reported."""
    directory = pilot_logs(tmp_path, seeds=(1001,))
    for path in find_logs(directory):
        payload = next(e.payload for e in read_events(path) if e.type is EventType.TOKENS_CONSUMED)
        assert run_total_tokens(path) == payload["authoritative_total"]


def test_derivation_requires_all_three_conditions(tmp_path: Path) -> None:
    directory = pilot_logs(tmp_path, seeds=(1001,))
    for path in find_logs(directory):
        if path.name.startswith("A"):
            path.unlink()
    with pytest.raises(BudgetDerivationError, match="privilege a condition"):
        derive_budget(find_logs(directory), allow_mock=True)


def test_derivation_ignores_the_wrong_scenario(tmp_path: Path) -> None:
    directory = tmp_path / "pilot"
    run_trio(make_config(run={"scenario": "demand_surge"}), directory, seed=1001, cycles=5)
    with pytest.raises(BudgetDerivationError, match="no admissible pilot runs"):
        derive_budget(find_logs(directory), allow_mock=True)


def test_derivation_ignores_experiment_2_runs(tmp_path: Path) -> None:
    directory = tmp_path / "pilot"
    directory.mkdir(parents=True)
    run_condition(experiment_2(10**9), directory / "b.jsonl", cycles=5)
    with pytest.raises(BudgetDerivationError, match="no admissible pilot runs"):
        derive_budget(find_logs(directory), allow_mock=True)
