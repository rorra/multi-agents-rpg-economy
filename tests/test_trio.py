"""Seed-paired A/B/C trios and the shared global view (design v1.0 s3.3, s12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpg_economy.domain import Condition, TokenBucket
from rpg_economy.engine import Engine, build_world
from rpg_economy.logging_ import EventLogWriter, EventType, read_events
from rpg_economy.orchestration.global_view import build_global_summary, render_global_summary
from rpg_economy.runner import run_trio
from tests.conftest import make_config, make_metadata

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
