"""Seed-paired A/B/C trios and the shared global view (design v1.0 s3.3, s12)."""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

from rpg_economy.domain import Condition, TokenBucket
from rpg_economy.engine import Engine, build_world
from rpg_economy.logging_ import EventLogWriter, EventType, log_digest, read_events
from rpg_economy.orchestration.global_view import (
    build_global_summary,
    render_global_summary,
)
from rpg_economy.runner import run_condition, run_trio
from tests.conftest import gold_flow, make_config, make_metadata

CYCLES = 40


def compressed() -> object:
    """A short run whose shock fits inside the observed window."""
    return make_config(
        phases={"base_ticks": 10, "shock_ticks": 15, "recovery_ticks": 10},
        time={"horizon_ticks": 35},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
    )


def events_of(path: Path) -> list:
    return list(read_events(path))


def shocks_of(path: Path) -> list[tuple[str, int]]:
    return [
        (e.type.value, e.tick)
        for e in read_events(path)
        if e.type in {EventType.SHOCK_APPLIED, EventType.SHOCK_REVERTED}
    ]


def expeditions_of(path: Path) -> list[tuple[str, str, int]]:
    """Expedition outcomes keyed by agent, ignoring when they were launched."""
    return [
        (e.agent_id, e.payload["outcome"], e.payload["total_units"])
        for e in read_events(path)
        if e.type is EventType.EXPEDITION_RESOLVED
    ]


# ---------------------------------------------------------------------------
# The shared global view (design v1.0 s3.3)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_the_director_view_and_the_bulletin_are_byte_identical(tmp_path: Path) -> None:
    """One generator, two consumers.

    If the Director's view in A and the bulletin in C could differ, a difference
    between the conditions would confound coordination with information -- which
    is the confound C exists to remove.
    """
    rendered: dict[Condition, str] = {}
    for condition in (Condition.A, Condition.C):
        config = make_config(run={"condition": condition.value})
        state = build_world(config)
        with EventLogWriter(tmp_path / f"{condition.value}.jsonl", make_metadata(config)) as log:
            engine = Engine(config, state, log)
            for _ in range(12):
                engine.run_cycle()
            rendered[condition] = render_global_summary(build_global_summary(config, engine))

    assert rendered[Condition.A] == rendered[Condition.C]
    assert rendered[Condition.A].encode() == rendered[Condition.C].encode()


@pytest.mark.invariant
def test_a_trios_first_director_view_is_its_first_bulletin_byte_for_byte(
    tmp_path: Path,
) -> None:
    """As the runs log them: A's cycle-0 view and C's cycle-0 bulletin.

    Nobody has acted before the first decision phase, so the two conditions
    hold the same world there, and what the Director reads must be exactly
    what C's agents are handed (design s3.3).
    """
    import hashlib

    runs = run_trio(compressed(), tmp_path, cycles=2)

    def first(condition: Condition, kind: EventType):
        return next(e for e in events_of(runs[condition].log_path) if e.type is kind)

    view = first(Condition.A, EventType.DIRECTOR_VIEW)
    bulletin = first(Condition.C, EventType.BULLETIN_PUBLISHED)
    assert view.cycle == bulletin.cycle == 0
    assert view.tick == bulletin.tick

    director_text = render_global_summary(view.payload["summary"])
    bulletin_text = render_global_summary(bulletin.payload["summary"])
    assert director_text.encode() == bulletin_text.encode()
    # The Director's own digest of what it read agrees with the bulletin's text.
    digest = hashlib.sha256(bulletin_text.encode("utf-8")).hexdigest()[:16]
    assert view.payload["view_sha"] == digest


def test_the_global_view_is_deterministic(tmp_path: Path) -> None:
    config = make_config()
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        for _ in range(10):
            engine.run_cycle()
        first = render_global_summary(build_global_summary(config, engine))
        second = render_global_summary(build_global_summary(config, engine))
    assert first == second


def test_the_global_view_reports_the_required_aggregates(tmp_path: Path) -> None:
    """Design s5.2: mean price, variation, volume, supply, unmet demand, shocks."""
    config = make_config()
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        summary = build_global_summary(config, engine)

    assert set(summary) >= {"version", "tick", "goods", "shock", "window"}
    good = summary["goods"][0]
    assert set(good) >= {
        "good",
        "mean_price",
        "price_variation",
        "volume_units",
        "volume_value",
        "available_supply",
        "unmet_demand",
    }


def test_the_global_view_exposes_no_private_holdings(tmp_path: Path) -> None:
    """Aggregates only: never an agent's gold, inventory or plans."""
    config = make_config()
    state = build_world(config)
    state.account("woodworking_1").gold = __import__("decimal").Decimal("9876.54")
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        rendered = render_global_summary(build_global_summary(config, engine))

    assert "9876.54" not in rendered
    assert "woodworking_1" not in rendered


def test_the_view_reports_an_active_shock(tmp_path: Path) -> None:
    config = make_config()
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        for _ in range(config.phases.shock_tick + 2):
            engine.run_cycle()
        summary = build_global_summary(config, engine)

    assert summary["shock"] is not None
    assert summary["shock"]["scenario"] == config.run.scenario.value


@pytest.mark.parametrize(
    ("scenario", "target"),
    [
        ("resource_scarcity", "target_resource"),
        ("demand_surge", "target_product"),
        ("productivity_drop", "target_specialization"),
        ("liquidity_loss", None),
    ],
)
def test_the_shock_line_shows_only_the_type_and_what_it_hit(
    tmp_path: Path, scenario: str, target: str | None
) -> None:
    """Never a multiplier, and never an aggregate of private gold.

    The view is shared by the Director (A) and the bulletin (C); a liquidity
    loss's frozen total would disclose private holdings to both.
    """
    config = make_config(run={"scenario": scenario})
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        while engine.shocks.active is None:
            engine.run_cycle()
        summary = build_global_summary(config, engine)
        rendered = render_global_summary(summary)

    expected = {"scenario"} | ({target} if target else set())
    assert set(summary["shock"]) == expected
    assert summary["shock"]["scenario"] == scenario
    shock_line = rendered.splitlines()[2]
    for hidden in ("multiplier", "frozen", "affected_agents", "fraction"):
        assert hidden not in shock_line


def test_a_good_that_did_not_trade_has_no_price_not_a_zero_one(tmp_path: Path) -> None:
    """A reader would take 0.00 for a price."""
    config = make_config()
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        summary = build_global_summary(config, engine)
        rendered = render_global_summary(summary)

    for good in summary["goods"]:
        assert good["trades"] == 0
        assert good["mean_price"] is None
        assert good["price_variation"] is None
    lines = rendered.splitlines()
    columns = lines[3].split("|")
    rows = [row.split("|") for row in lines[4:]]
    assert len(rows) == len(summary["goods"])
    for column in ("mean_price", "price_variation"):
        index = columns.index(column)
        assert all(row[index] == "-" for row in rows)


def test_price_variation_needs_two_trades(tmp_path: Path) -> None:
    from decimal import Decimal

    from rpg_economy.domain import OrderSide, Resource

    config = make_config()
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.state.account("warrior_1").receive(Resource.IRON, 5)

        def trade(price: str) -> None:
            engine.submit_order("warrior_1", OrderSide.SELL, Resource.IRON, 1, price)
            engine.submit_order("woodworking_1", OrderSide.BUY, Resource.IRON, 1, price)

        def iron() -> dict:
            summary = build_global_summary(config, engine)
            return next(g for g in summary["goods"] if g["good"] == "iron")

        trade("10")
        one = iron()
        trade("14")
        two = iron()

    assert one["trades"] == 1
    assert one["mean_price"] == "10.00"
    assert one["price_variation"] is None
    assert two["trades"] == 2
    assert Decimal(two["mean_price"]) == Decimal("12.00")
    assert two["price_variation"] is not None


def test_the_global_view_carries_its_legend(tmp_path: Path) -> None:
    """Part of the canonical text, so the Director and the bulletin share it."""
    config = make_config()
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        lines = render_global_summary(build_global_summary(config, engine)).splitlines()

    assert lines[0].startswith("global_view v")
    assert lines[1].startswith("columns: ")
    assert "- when too few trades" in lines[1]
    assert lines[2].startswith("shock: ")


def test_an_empty_view_still_carries_its_legend() -> None:
    rendered = render_global_summary(
        {"version": "x", "tick": 0, "cycle": 0, "window": 3, "goods": [], "shock": None}
    )
    assert rendered.splitlines()[1].startswith("columns: ")
    assert rendered.splitlines()[-1] == "goods: none"


# ---------------------------------------------------------------------------
# The bulletin in Condition C
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_only_condition_c_publishes_a_bulletin(tmp_path: Path) -> None:
    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=15)
    published = {
        condition: sum(1 for e in read_events(r.log_path) if e.type is EventType.BULLETIN_PUBLISHED)
        for condition, r in runs.items()
    }
    assert published[Condition.A] == 0
    assert published[Condition.B] == 0
    assert published[Condition.C] > 0


@pytest.mark.invariant
def test_bulletin_tokens_are_charged_only_in_condition_c(tmp_path: Path) -> None:
    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=CYCLES)
    assert runs[Condition.A].ledger.total(TokenBucket.BULLETIN) == 0
    assert runs[Condition.B].ledger.total(TokenBucket.BULLETIN) == 0
    assert runs[Condition.C].ledger.total(TokenBucket.BULLETIN) > 0


@pytest.mark.invariant
def test_the_bulletin_fires_no_trigger_in_a_full_run(tmp_path: Path) -> None:
    """Design s3.3: passive. It must not create activation demand of its own.

    Not only the same number of activations: the same triggers, fired for the
    same agents at the same ticks with the same payloads.
    """
    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=CYCLES)
    served = {
        condition: sum(1 for e in read_events(r.log_path) if e.type is EventType.ACTIVATION_SERVED)
        for condition in (Condition.B, Condition.C)
        for r in [runs[condition]]
    }
    assert served[Condition.B] == served[Condition.C]

    fired = {
        condition: [
            (e.tick, e.agent_id, e.payload)
            for e in read_events(runs[condition].log_path)
            if e.type is EventType.TRIGGER_FIRED
        ]
        for condition in (Condition.B, Condition.C)
    }
    assert fired[Condition.B], "no trigger fired; the comparison proves nothing"
    assert fired[Condition.B] == fired[Condition.C]


# ---------------------------------------------------------------------------
# Trio pairing (design v1.0 s12)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_trio_shares_its_shock_type_and_tick(tmp_path: Path) -> None:
    config = compressed()
    runs = run_trio(config, tmp_path, seed=1001, cycles=config.time.total_cycles)
    shocks = {c: shocks_of(r.log_path) for c, r in runs.items()}

    assert shocks[Condition.A] == shocks[Condition.B] == shocks[Condition.C]
    assert shocks[Condition.A] == [
        ("shock_applied", config.phases.shock_tick),
        ("shock_reverted", config.phases.reversion_tick),
    ]


@pytest.mark.invariant
def test_a_trio_shares_its_agents_and_endowments(tmp_path: Path) -> None:
    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=1)
    populations = {
        c: sorted(
            (a, r.state.account(a).role.value, r.state.account(a).specialization)
            for a in r.state.agent_ids
        )
        for c, r in runs.items()
    }
    assert populations[Condition.A] == populations[Condition.B] == populations[Condition.C]


@pytest.mark.invariant
def test_a_trio_shares_world_random_streams(tmp_path: Path) -> None:
    """Named streams mean an agent draws the same sequence in every condition.

    Activation *order* differs between A and B/C by design, so expeditions do
    not happen at the same ticks. What must match is the draw an agent gets on
    its n-th expedition, which is what makes the trio one world.
    """
    from rpg_economy.engine import RunRandom

    seeds = [RunRandom(1001), RunRandom(1001), RunRandom(1001)]
    draws = [[r.agent("warrior_1").random() for _ in range(6)] for r in seeds]
    assert draws[0] == draws[1] == draws[2]

    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=CYCLES)
    for condition, result in runs.items():
        assert result.config.run.seed == 1001, condition


def test_a_trio_differs_only_in_its_condition(tmp_path: Path) -> None:
    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=1)
    configs = {c: r.config.model_dump(mode="json") for c, r in runs.items()}

    a, b = configs[Condition.A], configs[Condition.B]
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"run"}
    assert a["run"]["seed"] == b["run"]["seed"]
    assert a["run"]["scenario"] == b["run"]["scenario"]


@pytest.mark.invariant
def test_each_condition_in_a_trio_is_reproducible(tmp_path: Path) -> None:
    first = run_trio(make_config(), tmp_path / "one", seed=1001, cycles=CYCLES)
    second = run_trio(make_config(), tmp_path / "two", seed=1001, cycles=CYCLES)

    for condition in Condition:
        assert log_digest(first[condition].log_path) == log_digest(second[condition].log_path), (
            condition
        )


def test_a_different_seed_gives_a_different_trio(tmp_path: Path) -> None:
    first = run_trio(make_config(), tmp_path / "one", seed=1001, cycles=CYCLES)
    second = run_trio(make_config(), tmp_path / "two", seed=2002, cycles=CYCLES)

    for condition in Condition:
        assert log_digest(first[condition].log_path) != log_digest(second[condition].log_path), (
            condition
        )


# ---------------------------------------------------------------------------
# Condition A end to end
# ---------------------------------------------------------------------------
def test_condition_a_runs_end_to_end_offline(tmp_path: Path) -> None:
    config = make_config(run={"condition": "A"})
    result = run_condition(config, tmp_path / "a.jsonl", cycles=CYCLES)
    seen = collections.Counter(e.type.value for e in read_events(result.log_path))

    for required in (
        EventType.DIRECTOR_CALL,
        EventType.DIRECTOR_TASK_ASSIGNED,
        EventType.RESULT_RETURNED,
        EventType.AGENT_OUTPUT,
        EventType.ACTION_EXECUTED,
    ):
        assert seen.get(required.value, 0) > 0, f"condition A never produced {required}"


@pytest.mark.invariant
def test_condition_a_never_exceeds_the_cap_over_a_run(tmp_path: Path) -> None:
    config = make_config(run={"condition": "A"})
    result = run_condition(config, tmp_path / "a.jsonl", cycles=CYCLES)
    cap = config.time.k_activations + 1

    for event in read_events(result.log_path):
        if event.type is EventType.DIRECTOR_CALL:
            assert event.payload["call_in_cycle"] <= cap
        if event.type is EventType.CYCLE_END:
            assert event.payload["effective_activations"] <= config.time.k_activations
            assert event.payload["director_calls"] <= cap


@pytest.mark.invariant
def test_all_three_conditions_conserve_gold(tmp_path: Path) -> None:
    expected = build_world(make_config()).total_gold()
    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=CYCLES)
    for condition, result in runs.items():
        assert result.state.total_gold() == expected + gold_flow(result.log_path), condition


@pytest.mark.invariant
def test_the_bulletin_is_published_when_the_director_would_read_its_view(tmp_path: Path) -> None:
    """After the cycle's mechanical ticks, before anyone acts (design s3.3)."""
    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=10)
    events = list(read_events(runs[Condition.C].log_path))
    for cycle in range(10):
        in_cycle = [e for e in events if e.cycle == cycle]
        kinds = [e.type for e in in_cycle]
        bulletin = kinds.index(EventType.BULLETIN_PUBLISHED)
        last_tick = max(i for i, kind in enumerate(kinds) if kind is EventType.TICK)
        assert last_tick < bulletin
        assert all(kind is not EventType.AGENT_OUTPUT for kind in kinds[:bulletin]), (
            f"an agent acted before cycle {cycle}'s bulletin"
        )


@pytest.mark.invariant
def test_a_provider_failure_ends_the_run_instead_of_becoming_waits(tmp_path) -> None:
    """An outage is not an agent's decision; the run must not complete around it."""
    from rpg_economy.llm.client import ProviderError
    from rpg_economy.logging_ import log_is_finished
    from rpg_economy.runner import run_condition
    from tests.conftest import make_config

    class Down:
        model = "down"

        def complete(self, *, system: str, user: str):
            raise ProviderError("Error code: 401 - invalid_api_key")

    for condition in ("A", "B"):
        path = tmp_path / f"{condition}.jsonl"
        with pytest.raises(ProviderError):
            run_condition(make_config(run={"condition": condition}), path, client=Down(), cycles=10)
        assert not log_is_finished(path)


# ---------------------------------------------------------------------------
# Two-level stop: abort a run mid-way, or stop a trio between conditions
# ---------------------------------------------------------------------------
def test_an_abort_mid_run_leaves_an_unfinished_log(tmp_path: Path) -> None:
    from rpg_economy.logging_ import log_is_finished
    from rpg_economy.runner import INTERRUPT_ABORT, RunInterrupted

    config = make_config(run={"condition": "B"})
    asked = []

    def should_abort() -> bool:
        asked.append(True)
        return len(asked) > 3  # three cycles run, the fourth is refused

    path = tmp_path / "b.jsonl"
    with pytest.raises(RunInterrupted) as caught:
        run_condition(config, path, cycles=10, should_abort=should_abort)

    assert caught.value.reason == INTERRUPT_ABORT
    assert caught.value.interrupted is Condition.B
    assert caught.value.completed == {}
    assert caught.value.log_path == path
    assert not log_is_finished(path)
    events = events_of(path)
    assert EventType.RUN_END not in {e.type for e in events}
    assert events[-1].type is EventType.RUN_ABORTED
    aborted = events[-1].payload
    assert aborted["cycles_completed"] == 3
    assert aborted["cycles_planned"] == 10
    assert sum(e.type is EventType.CYCLE_END for e in events) == 3
    # The spend so far is in the aborted run's own log, in tokens_consumed's
    # shape: the ledger snapshot and the budget's.
    assert {"grand_total", "authoritative_total", "billed_total", "budget"} <= aborted.keys()
    assert aborted["budget"]["cap"] is None


def test_run_aborted_carries_what_tokens_consumed_carries(tmp_path: Path) -> None:
    from rpg_economy.runner import RunInterrupted

    config = make_config(run={"condition": "B"})
    cycles = 6
    finished = run_condition(config, tmp_path / "done.jsonl", cycles=cycles)
    consumed = next(
        e for e in events_of(finished.log_path) if e.type is EventType.TOKENS_CONSUMED
    ).payload
    assert consumed["grand_total"] > 0, "nothing was spent; the comparison proves nothing"

    asked = []
    with pytest.raises(RunInterrupted):
        run_condition(
            config,
            tmp_path / "cut.jsonl",
            cycles=cycles + 4,
            should_abort=lambda: asked.append(True) or len(asked) > cycles,
        )
    aborted = events_of(tmp_path / "cut.jsonl")[-1].payload
    # The same cycles run: the same spend, with the cycle counts beside it.
    assert aborted["cycles_completed"] == cycles
    assert {k: v for k, v in aborted.items() if not k.startswith("cycles_")} == consumed


def test_run_interrupted_has_a_short_repr() -> None:
    from rpg_economy.runner import INTERRUPT_ABORT, RunInterrupted

    exc = RunInterrupted(reason=INTERRUPT_ABORT, interrupted=Condition.B)
    assert repr(exc) == "RunInterrupted(reason='abort', interrupted='B')"
    assert str(exc) == "run interrupted (abort during B); finished: none"
    # eq=False: two interruptions are two events, never equal by value.
    assert exc != RunInterrupted(reason=INTERRUPT_ABORT, interrupted=Condition.B)


def test_a_run_without_an_abort_callback_is_unchanged(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    plain = run_condition(config, tmp_path / "plain.jsonl", cycles=5)
    asked = run_condition(config, tmp_path / "asked.jsonl", cycles=5, should_abort=lambda: False)
    assert log_digest(plain.log_path) == log_digest(asked.log_path)
    assert EventType.RUN_ABORTED not in {e.type for e in events_of(asked.log_path)}


def test_a_stop_between_conditions_keeps_the_finished_ones(tmp_path: Path) -> None:
    from rpg_economy.logging_ import log_is_finished
    from rpg_economy.runner import INTERRUPT_STOP, RunInterrupted

    asked = []

    def should_stop() -> bool:
        asked.append(True)
        return len(asked) > 2  # A and B start; C does not

    with pytest.raises(RunInterrupted) as caught:
        run_trio(make_config(), tmp_path, seed=1001, cycles=3, should_stop=should_stop)

    assert caught.value.reason == INTERRUPT_STOP
    assert caught.value.interrupted is None
    assert set(caught.value.completed) == {Condition.A, Condition.B}
    for result in caught.value.completed.values():
        assert log_is_finished(result.log_path)
    assert not (tmp_path / "C.jsonl").exists()


def test_an_abort_inside_a_trio_keeps_the_conditions_already_finished(tmp_path: Path) -> None:
    from rpg_economy.logging_ import log_is_finished
    from rpg_economy.runner import INTERRUPT_ABORT, RunInterrupted

    asked = []

    def should_abort() -> bool:
        asked.append(True)
        return len(asked) > 5  # A's three cycles, then two of B's

    with pytest.raises(RunInterrupted) as caught:
        run_trio(make_config(), tmp_path, seed=1001, cycles=3, should_abort=should_abort)

    assert caught.value.reason == INTERRUPT_ABORT
    assert caught.value.interrupted is Condition.B
    assert set(caught.value.completed) == {Condition.A}
    assert log_is_finished(caught.value.completed[Condition.A].log_path)
    assert not log_is_finished(tmp_path / "B.jsonl")
    assert not (tmp_path / "C.jsonl").exists()


@pytest.mark.parametrize("phase", ["pilot", "final", "adhoc"])
def test_the_run_phase_is_recorded_in_the_metadata(tmp_path: Path, phase: str) -> None:
    from rpg_economy.logging_ import read_metadata

    config = make_config(run={"condition": "B", "phase": phase})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=1)
    assert read_metadata(result.log_path).phase == phase


def test_a_log_without_a_phase_still_reads() -> None:
    from rpg_economy.logging_ import RunMetadata

    assert RunMetadata.model_fields["phase"].default is None
