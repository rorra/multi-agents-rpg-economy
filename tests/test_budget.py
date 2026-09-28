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
from rpg_economy.domain import Condition, TokenBucket
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


def test_the_derivation_records_its_evidence(tmp_path: Path) -> None:
    """Recorded in the config, never hand-tuned (design s8.4, s15)."""
    derivation = derive_budget(find_logs(pilot_logs(tmp_path)), allow_mock=True)
    payload = derivation.to_payload()

    assert payload["scenario"] == "resource_scarcity"
    assert "pooled median" in payload["rule"]
    assert set(payload["per_condition_totals"]) == {"A", "B", "C"}
    assert payload["pooled_totals"] == sorted(payload["pooled_totals"])


@pytest.mark.invariant
def test_experiment_2_refuses_to_run_without_a_derived_budget() -> None:
    from rpg_economy.config import ConfigError

    with pytest.raises(ConfigError, match="experiment_2_total_tokens"):
        make_config(run={"experiment": "experiment_2"})


# ---------------------------------------------------------------------------
# The Experiment 2 trio gate
# ---------------------------------------------------------------------------
def measured_budget(tmp_path: Path, cycles: int, *, fraction: float = 0.35) -> int:
    """A budget set from what an unbudgeted run of the same shape actually spends.

    A token literal here silently stops testing anything the moment prompt sizes
    or K change, since consumption can fall below the literal and nothing
    exhausts. Deriving it keeps the exhaustion tests about exhaustion.
    """
    reference = run_trio(
        make_config(run={"experiment": "experiment_1"}),
        tmp_path / "_measure",
        seed=1001,
        cycles=cycles,
    )
    return max(1, int(reference[Condition.A].ledger.authoritative_total * fraction))


def experiment_2_trio(tmp_path: Path, cycles: int, budget: int | None = None) -> dict:
    """An Experiment 2 trio at a budget chosen so A exhausts and B/C need not."""
    if budget is None:
        budget = measured_budget(tmp_path, cycles)
    base = make_config(
        run={"experiment": "experiment_2"},
        budget={"experiment_2_total_tokens": budget},
    )
    return run_trio(base, tmp_path / "_trio", seed=1001, cycles=cycles)


@pytest.mark.invariant
def test_condition_a_exhausts_first_under_a_shared_budget(tmp_path: Path) -> None:
    """A carries Director overhead on top of the same agent work as B and C."""
    runs = experiment_2_trio(tmp_path, cycles=120)

    assert runs[Condition.A].budget.exhausted, "A did not exhaust; raise the cycles"
    a_tick = runs[Condition.A].exhaustion_tick
    for condition in (Condition.B, Condition.C):
        other = runs[condition].exhaustion_tick
        assert other is None or other > a_tick, (
            f"{condition.value} exhausted at {other}, not after A at {a_tick}"
        )


@pytest.mark.invariant
def test_the_exhausted_condition_makes_no_calls_afterwards(tmp_path: Path) -> None:
    runs = experiment_2_trio(tmp_path, cycles=120)
    result = runs[Condition.A]
    assert result.budget.exhausted

    events = list(read_events(result.log_path))
    exhausted = next(e for e in events if e.type is EventType.BUDGET_EXHAUSTED)
    after = [e for e in events if e.seq > exhausted.seq]

    # No call of any kind is made once the budget is gone.
    assert not [e for e in after if e.type is EventType.LLM_CALL]
    assert not [e for e in after if e.type is EventType.DIRECTOR_CALL]

    # The activation in flight when the cap was crossed still finishes. Its call
    # was authorised and paid for before exhaustion; throwing the answer away
    # would spend the tokens and get nothing for them. Exactly one such
    # activation can exist, since activations run sequentially.
    outputs_after = [e for e in after if e.type is EventType.AGENT_OUTPUT]
    assert len(outputs_after) <= 1, (
        f"{len(outputs_after)} activations ran after exhaustion; at most the one "
        f"already in flight may finish"
    )


@pytest.mark.invariant
def test_the_exhausted_condition_keeps_ticking(tmp_path: Path) -> None:
    """Design v1.0 s8.4: mechanical ticks continue to the planned horizon.

    The budget is a fraction of what the run would spend unbudgeted, rather than
    a fixed token count: a literal here silently stops testing anything the
    moment prompt sizes change.
    """
    cycles = 120
    runs = experiment_2_trio(tmp_path, cycles=cycles)
    result = runs[Condition.A]

    events = list(read_events(result.log_path))
    exhausted = next(e for e in events if e.type is EventType.BUDGET_EXHAUSTED)
    ticks_after = [e for e in events if e.seq > exhausted.seq and e.type is EventType.TICK]

    # The invariant is reaching the planned horizon, not any particular margin.
    assert result.state.tick == cycles
    assert len(ticks_after) > 10, "the run stopped ticking after exhaustion"
    cycles_after = [e for e in events if e.seq > exhausted.seq and e.type is EventType.CYCLE_END]
    assert cycles_after
    assert all(e.payload["effective_activations"] == 0 for e in cycles_after[1:])


@pytest.mark.invariant
def test_all_three_conditions_reach_the_same_planned_horizon(tmp_path: Path) -> None:
    """Primary metrics are computed over the full horizon for every condition."""
    cycles = 120
    runs = experiment_2_trio(tmp_path, cycles=cycles)

    ticks = {condition: result.state.tick for condition, result in runs.items()}
    assert len(set(ticks.values())) == 1, ticks
    assert ticks[Condition.A] == cycles


def test_the_trio_reports_its_common_active_horizon(tmp_path: Path) -> None:
    """Design v1.0 s8.4: the secondary analysis interval for Experiment 2."""
    runs = experiment_2_trio(tmp_path, cycles=120)
    horizon = trio_common_active_horizon(runs)

    exhaustions = [r.exhaustion_tick for r in runs.values() if r.exhaustion_tick is not None]
    assert exhaustions
    assert horizon == min(exhaustions)
    assert horizon < make_config().time.horizon_ticks


def test_a_trio_that_never_exhausts_uses_the_full_horizon(tmp_path: Path) -> None:
    runs = experiment_2_trio(tmp_path, budget=10**12, cycles=20)
    assert not any(r.budget.exhausted for r in runs.values())
    assert trio_common_active_horizon(runs) == make_config().time.horizon_ticks


@pytest.mark.invariant
def test_an_exhausted_trio_still_passes_the_integrity_gate(tmp_path: Path) -> None:
    """Exhaustion must not look like a condition that failed to take effect."""
    from rpg_economy.integrity import verify_trio

    runs = experiment_2_trio(tmp_path, cycles=120)
    report = verify_trio(runs)
    assert "director_tokens_present_in_a" in report.checks_passed
    assert "bulletin_tokens_present_in_c" in report.checks_passed


@pytest.mark.invariant
@pytest.mark.parametrize("condition", ["A", "B", "C"])
def test_no_activation_starts_after_exhaustion(tmp_path: Path, condition: str) -> None:
    """Design s8.4: once the budget is spent, later activations are refused.

    Holds in every condition. The activation in flight when the cap is crossed
    may finish; nothing may start after it, or B and C would log a string of
    budget WAITs that A, whose Director stops calling, never does.
    """
    config = experiment_2(150_000, run={"condition": condition})
    client = InflatedClient(json_action(action="wait"))
    result = run_condition(config, tmp_path / "run.jsonl", client=client, cycles=60)

    events = list(read_events(result.log_path))
    exhausted = next(e for e in events if e.type is EventType.BUDGET_EXHAUSTED)
    started = (
        EventType.ACTIVATION_SERVED,
        EventType.OBSERVATION_DELIVERED,
        EventType.DIRECTOR_TASK_ASSIGNED,
    )
    after = [e for e in events if e.seq > exhausted.seq and e.type in started]
    assert after == [], f"{len(after)} activations started after exhaustion in {condition}"


@pytest.mark.parametrize("condition", ["B", "C"])
def test_requests_left_unserved_by_the_budget_are_logged_as_deferred(
    tmp_path: Path, condition: str
) -> None:
    """Design s15: every request is accounted for, with why it was not served."""
    config = experiment_2(150_000, run={"condition": condition})
    client = InflatedClient(json_action(action="wait"))
    result = run_condition(config, tmp_path / "run.jsonl", client=client, cycles=60)

    events = list(read_events(result.log_path))
    exhausted = next(e for e in events if e.type is EventType.BUDGET_EXHAUSTED)
    deferred = [e for e in events if e.type is EventType.ACTIVATION_DEFERRED]
    for_budget = [e for e in deferred if e.payload["reason"] == "budget_exhausted"]
    assert for_budget, "no request was deferred for the budget; the test proves nothing"
    assert all(e.seq > exhausted.seq for e in for_budget)
    assert {e.payload["reason"] for e in deferred} <= {"budget_exhausted", "k_limit"}
    assert all(e.payload["reason"] == "k_limit" for e in deferred if e.seq < exhausted.seq)

    # Every post-exhaustion cycle defers what is pending, and serves nothing.
    ends = [e for e in events if e.type is EventType.CYCLE_END and e.payload.get("post_exhaustion")]
    assert ends
    for end in ends:
        in_cycle = [e for e in for_budget if e.cycle == end.cycle]
        assert len(in_cycle) == end.payload["deferred"]


# ---------------------------------------------------------------------------
# What the derivation refuses (design v1.0 s8.4)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_mock_runs_cannot_set_the_budget(tmp_path: Path) -> None:
    with pytest.raises(BudgetDerivationError, match="mock run"):
        derive_budget(find_logs(pilot_logs(tmp_path, seeds=(1001,))))


def test_a_run_counted_twice_is_refused(tmp_path: Path) -> None:
    directory = pilot_logs(tmp_path, seeds=(1001,))
    run_trio(make_config(), tmp_path / "again", seed=1001, cycles=6)
    with pytest.raises(BudgetDerivationError, match="appears twice"):
        derive_budget(find_logs(tmp_path), allow_mock=True)
    assert directory.exists()


def test_a_seed_missing_a_condition_is_refused(tmp_path: Path) -> None:
    directory = pilot_logs(tmp_path, seeds=(1001, 1002))
    next(p for p in find_logs(directory / "1002") if p.name == "B.jsonl").unlink()
    with pytest.raises(BudgetDerivationError, match="seed 1002"):
        derive_budget(find_logs(directory), allow_mock=True)


@pytest.mark.invariant
def test_runs_shorter_than_the_full_horizon_are_refused(tmp_path: Path) -> None:
    """The calibration stage runs shorter; its totals would understate the budget."""
    directory = pilot_logs(tmp_path, seeds=(1001,))
    with pytest.raises(BudgetDerivationError, match="not the full"):
        derive_budget(find_logs(directory), config=make_config(), allow_mock=True)


def test_a_final_seed_cannot_set_the_budget(tmp_path: Path) -> None:
    config = make_config()
    final_seed = config.seeds.final[0]
    run_trio(config, tmp_path / "final", seed=final_seed, cycles=6)
    with pytest.raises(BudgetDerivationError, match="not a pilot seed"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


def test_a_whole_pilot_at_the_full_horizon_is_admitted(tmp_path: Path) -> None:
    short = {
        "phases": {"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 4},
        "time": {"horizon_ticks": 12},
        "metrics": {"volatility_window_w": 2, "recovery_hold_ticks_m": 2},
    }
    # Only runs recorded as the pilot's set the budget.
    config = make_config(
        **short, run={"phase": "pilot"}, seeds={"pilot": [1001, 1002], "final": [2001, 2002]}
    )
    from rpg_economy.integrity import verify_trio_logs, write_integrity_marker

    for seed in (1001, 1002):
        run_trio(config, tmp_path / str(seed), seed=seed)
        # Only trios that passed the integrity gate set the budget, as a pilot's do.
        write_integrity_marker(tmp_path / str(seed), verify_trio_logs(tmp_path / str(seed)))
    derivation = derive_budget(find_logs(tmp_path), config=config, allow_mock=True)
    assert len(derivation.runs) == 6


def test_after_exhaustion_resolved_requests_are_dropped_not_deferred(tmp_path: Path) -> None:
    """A request whose triggers stopped holding is resolved, as on the ordinary
    path, rather than logged as deferred every cycle until the horizon."""
    from rpg_economy.domain import Trigger
    from rpg_economy.engine import Engine, build_world
    from rpg_economy.logging_ import EventLogWriter
    from rpg_economy.orchestration import DecentralizedOrchestrator
    from rpg_economy.orchestration.activation import ActivationRequest
    from tests.conftest import make_metadata

    config = make_config(run={"condition": "B"})
    budget = TokenBudget(cap=10)
    budget.charge(10, tick=0, cycle=0)
    assert not budget.may_call()

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        orchestrator = DecentralizedOrchestrator(
            config, engine, lambda e, a: pytest.fail("activated after exhaustion"), budget=budget
        )
        triggers = orchestrator.triggers
        triggers.clear_pending()
        # T1 with an empty inbox no longer holds: the request is resolved.
        triggers._pending["woodworking_1"] = ActivationRequest(
            agent_id="woodworking_1",
            first_requested_tick=0,
            triggers={Trigger.T1_MESSAGE_RECEIVED},
        )
        orchestrator(engine)
        summary = dict(orchestrator._last_cycle)

    events = list(read_events(log.path))
    assert not [e for e in events if e.type is EventType.ACTIVATION_DEFERRED]
    resolved = [e for e in events if e.type is EventType.ACTIVATION_REQUEST_RESOLVED]
    assert [e.agent_id for e in resolved] == ["woodworking_1"]
    assert summary["post_exhaustion"] is True
    assert summary["requested"] == summary["deferred"] == 0
    assert triggers.pending == {}
