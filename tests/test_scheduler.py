"""The neutral scheduler and the K activation budget (design v1.0 s8.1, s8.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpg_economy.engine import Engine, build_world
from rpg_economy.engine.rng import RunRandom
from rpg_economy.logging_ import EventLogWriter, EventType, read_events
from rpg_economy.orchestration import DecentralizedOrchestrator, NeutralScheduler
from tests.conftest import make_config, make_metadata

AGENTS = [f"agent_{i}" for i in range(8)]


def scheduler(k: int = 3, seed: int = 7, agents: list[str] | None = None) -> NeutralScheduler:
    return NeutralScheduler(agents or AGENTS, RunRandom(seed), k)


# ---------------------------------------------------------------------------
# K budget
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_never_serves_more_than_k(scheduler_k: int = 3) -> None:
    sched = scheduler(k=scheduler_k)
    assert len(sched.serve(AGENTS)) == scheduler_k


def test_serves_everyone_when_requests_fit_within_k() -> None:
    sched = scheduler(k=len(AGENTS))
    assert sorted(sched.serve(AGENTS)) == sorted(AGENTS)


def test_serves_nothing_when_nothing_requests() -> None:
    assert scheduler().serve([]) == []


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="K must be at least 1"):
        scheduler(k=0)


# ---------------------------------------------------------------------------
# Priority: fewest accumulated activations
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_priority_goes_to_the_least_activated() -> None:
    sched = scheduler(k=2)
    for _ in range(5):
        sched.record_activation("agent_0")
    for _ in range(3):
        sched.record_activation("agent_1")

    served = sched.serve(["agent_0", "agent_1", "agent_2", "agent_3"])
    assert "agent_0" not in served
    assert "agent_1" not in served


def test_repeated_service_equalizes_activations() -> None:
    """The rule exists so no agent is starved over a run."""
    sched = scheduler(k=3)
    for _ in range(40):
        for agent_id in sched.serve(AGENTS):
            sched.record_activation(agent_id)

    counts = [sched.activations_of(a) for a in AGENTS]
    assert max(counts) - min(counts) <= 1


def test_activations_are_counted_only_when_recorded() -> None:
    sched = scheduler()
    sched.serve(AGENTS)
    assert sched.total_activations == 0

    for agent_id in sched.serve(AGENTS):
        sched.record_activation(agent_id)
    assert sched.total_activations == sched.k


# ---------------------------------------------------------------------------
# Tie-break: fixed permutation from the run seed
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_tie_break_is_derived_from_the_seed_and_is_stable() -> None:
    assert scheduler(seed=7).tie_break_order == scheduler(seed=7).tie_break_order
    assert scheduler(seed=7).tie_break_order != scheduler(seed=8).tie_break_order


def test_tie_break_is_a_permutation_of_all_agents() -> None:
    assert sorted(scheduler().tie_break_order) == sorted(AGENTS)


def test_ties_are_broken_by_that_permutation() -> None:
    sched = scheduler(k=2)
    order = sched.tie_break_order
    # Every agent has zero activations, so only the permutation decides.
    assert sched.serve(AGENTS) == order[:2]


def test_the_tie_break_does_not_drift_during_a_run() -> None:
    """It is drawn once; a permutation that moved would be a hidden variable."""
    sched = scheduler(k=2)
    before = sched.tie_break_order
    for _ in range(20):
        for agent_id in sched.serve(AGENTS):
            sched.record_activation(agent_id)
    assert sched.tie_break_order == before


def test_service_order_is_the_order_agents_will_act() -> None:
    sched = scheduler(k=3)
    served = sched.serve(AGENTS)
    ranks = [sched.tie_break_order.index(a) for a in served]
    assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# Neutrality
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_request_order_cannot_influence_the_outcome() -> None:
    """Neutrality: the scheduler must not privilege whoever asked first."""
    sched_a, sched_b = scheduler(k=3), scheduler(k=3)
    assert sched_a.serve(AGENTS) == sched_b.serve(list(reversed(AGENTS)))


def test_duplicate_requests_do_not_buy_extra_turns() -> None:
    sched = scheduler(k=3)
    assert sched.serve(AGENTS) == sched.serve(AGENTS + AGENTS)


def test_an_unknown_agent_is_rejected() -> None:
    with pytest.raises(KeyError, match="does not know"):
        scheduler().serve(["stranger"])


# ---------------------------------------------------------------------------
# End to end, inside a run
# ---------------------------------------------------------------------------
@pytest.fixture
def orchestrated(tmp_path: Path):
    config = make_config(run={"condition": "B"})
    state = build_world(config)
    activated: list[str] = []
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        orchestrator = DecentralizedOrchestrator(config, engine, lambda e, a: activated.append(a))
        yield config, engine, orchestrator, activated


@pytest.mark.invariant
def test_a_cycle_never_activates_more_than_k(orchestrated) -> None:
    config, engine, orchestrator, activated = orchestrated
    for _ in range(30):
        before = len(activated)
        engine.run_cycle(orchestrator)
        assert len(activated) - before <= config.time.k_activations


def test_effective_activations_are_reported_at_cycle_end(orchestrated) -> None:
    config, engine, orchestrator, activated = orchestrated
    for _ in range(10):
        engine.run_cycle(orchestrator)
    engine.log.close()

    ends = [e for e in read_events(engine.log.path) if e.type is EventType.CYCLE_END]
    assert len(ends) == 10
    for event in ends:
        assert "effective_activations" in event.payload
        assert event.payload["effective_activations"] <= config.time.k_activations


def test_exactly_one_cycle_end_per_macro_cycle(orchestrated) -> None:
    _, engine, orchestrator, _ = orchestrated
    for _ in range(12):
        engine.run_cycle(orchestrator)
    engine.log.close()

    ends = [e for e in read_events(engine.log.path) if e.type is EventType.CYCLE_END]
    assert [e.cycle for e in ends] == list(range(12))


def test_deferred_requests_are_logged(orchestrated) -> None:
    config, engine, orchestrator, _ = orchestrated
    for _ in range(40):
        engine.run_cycle(orchestrator)
    engine.log.close()

    events = list(read_events(engine.log.path))
    served = [e for e in events if e.type is EventType.ACTIVATION_SERVED]
    assert served
    for event in served:
        assert "triggers" in event.payload
        assert "waiting_ticks" in event.payload


def test_a_cycle_with_no_requests_closes_early(tmp_path: Path) -> None:
    """Design v1.0 s8.1: a macro-cycle may end before exhausting K.

    Triggers are cleared at the cycle boundary, after the mechanical ticks have
    run: clearing them earlier would simply see them fire again during the same
    cycle's ticks, and the test would prove nothing.
    """
    config = make_config(run={"condition": "B"})
    state = build_world(config)
    activated: list[str] = []

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        orchestrator = DecentralizedOrchestrator(config, engine, lambda e, a: activated.append(a))

        def quiet_cycle(e: Engine) -> None:
            orchestrator.triggers.clear_pending()
            orchestrator(e)

        engine.run_cycle(orchestrator)
        before = orchestrator.cycles_closed_early
        served_before = len(activated)
        engine.run_cycle(quiet_cycle)
        engine.log.close()

    assert orchestrator.cycles_closed_early == before + 1
    assert len(activated) == served_before

    ends = [e for e in read_events(engine.log.path) if e.type is EventType.CYCLE_END]
    assert ends[-1].payload["closed_early"] is True
    assert ends[-1].payload["effective_activations"] == 0
    # One schema for every branch: the early close carries post_exhaustion too.
    assert ends[-1].payload["post_exhaustion"] is False
    assert {"post_exhaustion", "closed_early"} <= set(ends[0].payload)


def test_an_unserved_agent_is_activated_in_a_later_cycle(orchestrated) -> None:
    """Persistence has to translate into eventual service, not just a standing request."""
    config, engine, orchestrator, activated = orchestrated
    for _ in range(60):
        engine.run_cycle(orchestrator)

    served_counts = {a: orchestrator.scheduler.activations_of(a) for a in engine.state.agent_ids}
    assert all(count > 0 for count in served_counts.values()), served_counts


@pytest.mark.invariant
def test_the_orchestrator_hands_the_scheduler_sorted_ids_only(orchestrated) -> None:
    """Neutrality is structural: serve() receives ids, never the request objects."""
    config, engine, orchestrator, activated = orchestrated
    received: list[object] = []
    serve = orchestrator.scheduler.serve

    def spy(requesting):
        received.append(requesting)
        return serve(requesting)

    orchestrator.scheduler.serve = spy
    for _ in range(30):
        engine.run_cycle(orchestrator)

    assert received, "the scheduler was never asked; the test proves nothing"
    for requesting in received:
        assert isinstance(requesting, list)
        assert all(isinstance(agent_id, str) for agent_id in requesting)
        assert requesting == sorted(requesting)
