"""Shocks: finite duration, exact timing, and per-scenario mechanisms."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from rpg_economy.agents.observation import agent_status, confiscation_lines
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import OrderSide, Resource, Role, Scenario
from rpg_economy.engine import Engine, build_world
from rpg_economy.logging_ import EventLogWriter, EventType, read_events
from tests.conftest import make_config, make_metadata

IRON = Resource.IRON


def run_scenario(path: Path, config: SimulationConfig, cycles: int) -> Engine:
    state = build_world(config)
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        for _ in range(cycles):
            engine.run_cycle()
    return engine


def short_config(scenario: Scenario, *, seed: int | None = None) -> SimulationConfig:
    """A compressed run: 10 base ticks, 5 of shock, 10 of recovery.

    W shrinks with the base phase: the recovery reference is the mean over the
    last W base-phase ticks (design s11.1), so a window wider than the
    phase is a contradiction the schema rejects.
    """
    run: dict[str, object] = {"scenario": scenario.value}
    if seed is not None:
        run["seed"] = seed
    return make_config(
        run=run,
        phases={"base_ticks": 10, "shock_ticks": 5, "recovery_ticks": 10},
        time={"horizon_ticks": 25},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
    )


# ---------------------------------------------------------------------------
# Timing (design v1.0 s10)
# ---------------------------------------------------------------------------
def test_shock_tick_and_reversion_tick_derive_from_the_phases() -> None:
    config = short_config(Scenario.RESOURCE_SCARCITY)
    assert config.phases.shock_tick == 10
    assert config.phases.reversion_tick == 15


def test_shock_fires_and_reverts_at_the_configured_ticks(tmp_path: Path) -> None:
    config = short_config(Scenario.RESOURCE_SCARCITY)
    run_scenario(tmp_path / "run.jsonl", config, cycles=25)

    events = [
        (e.type, e.tick)
        for e in read_events(tmp_path / "run.jsonl")
        if e.type in {EventType.SHOCK_APPLIED, EventType.SHOCK_REVERTED}
    ]
    assert events == [
        (EventType.SHOCK_APPLIED, config.phases.shock_tick),
        (EventType.SHOCK_REVERTED, config.phases.reversion_tick),
    ]


def test_exactly_one_main_shock_per_run(tmp_path: Path) -> None:
    """Design v1.0 s10: one base phase, one shock, one recovery phase."""
    config = short_config(Scenario.RESOURCE_SCARCITY)
    run_scenario(tmp_path / "run.jsonl", config, cycles=25)

    applied = [e for e in read_events(tmp_path / "run.jsonl") if e.type is EventType.SHOCK_APPLIED]
    assert len(applied) == 1


def test_no_shock_before_its_tick(tmp_path: Path) -> None:
    config = short_config(Scenario.RESOURCE_SCARCITY)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=10)
    assert engine.shocks.active is None
    assert engine.zones.spawn_multiplier(IRON) == Decimal(1)


def test_the_shock_is_active_between_its_ticks(tmp_path: Path) -> None:
    config = short_config(Scenario.RESOURCE_SCARCITY)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=12)
    assert engine.shocks.active is not None
    assert engine.shocks.active.scenario is Scenario.RESOURCE_SCARCITY


def test_timing_is_identical_for_any_seed(tmp_path: Path) -> None:
    """A seed-trio must receive the same shock at the same tick (design s10)."""
    ticks = []
    for seed in (1, 2, 3):
        config = short_config(Scenario.RESOURCE_SCARCITY, seed=seed)
        run_scenario(tmp_path / f"{seed}.jsonl", config, cycles=25)
        ticks.append(
            [
                e.tick
                for e in read_events(tmp_path / f"{seed}.jsonl")
                if e.type in {EventType.SHOCK_APPLIED, EventType.SHOCK_REVERTED}
            ]
        )
    assert ticks[0] == ticks[1] == ticks[2]


# ---------------------------------------------------------------------------
# resource_scarcity -- the primary scenario
# ---------------------------------------------------------------------------
def test_resource_scarcity_reduces_the_targets_appearance(tmp_path: Path) -> None:
    config = short_config(Scenario.RESOURCE_SCARCITY)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=12)

    target = Resource(config.shocks.resource_scarcity.target_resource)
    assert engine.zones.spawn_multiplier(target) == config.shocks.resource_scarcity.spawn_multiplier
    assert engine.zones.spawn_multiplier(Resource.WOOD) == Decimal(1)


def test_resource_scarcity_can_take_the_stock_already_in_the_pools(tmp_path: Path) -> None:
    """Pools built up while a zone went unvisited would otherwise cover the shock."""
    base = short_config(Scenario.RESOURCE_SCARCITY)
    target = Resource(base.shocks.resource_scarcity.target_resource)
    config = make_config(
        run={"scenario": Scenario.RESOURCE_SCARCITY.value},
        phases={"base_ticks": 10, "shock_ticks": 5, "recovery_ticks": 10},
        time={"horizon_ticks": 25},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
        shocks={"resource_scarcity": {"stock_multiplier": "0"}},
    )
    before = run_scenario(tmp_path / "before.jsonl", base, cycles=10).zones.total_available(target)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=11)

    assert before > 1
    # What remains is at most the one tick of reduced appearance since.
    assert engine.zones.total_available(target) <= 1
    applied = next(
        e for e in read_events(tmp_path / "run.jsonl") if e.type is EventType.SHOCK_APPLIED
    )
    assert applied.payload["stock_multiplier"] == "0"


def test_resource_scarcity_takes_held_stock_over_the_cap_cutting_sell_orders(
    tmp_path: Path,
) -> None:
    """A stock gathered before the shock would otherwise sell cheap through it."""
    config = make_config(
        run={"scenario": Scenario.RESOURCE_SCARCITY.value},
        phases={"base_ticks": 10, "shock_ticks": 5, "recovery_ticks": 10},
        time={"horizon_ticks": 25},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
        shocks={"resource_scarcity": {"held_keep": 3}},
    )
    target = Resource(config.shocks.resource_scarcity.target_resource)
    state = build_world(config)
    first, second = state.agent_ids[:2]
    for agent_id in state.agent_ids:  # some start with a little of it
        account = state.account(agent_id)
        account.consume(target, account.quantity(target))
    state.account(first).receive(target, 5)
    state.account(second).receive(target, 4)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.market.place(
            agent_id=second, side=OrderSide.SELL, good=target, quantity=3, price=Decimal("9"), tick=0
        )
        shock = engine.shocks.apply(config.phases.shock_tick)

    # Nine held, three kept: one at a time from the largest holder, ties in
    # agent order, so the first gives up 4 and the second 2.
    price = Decimal(str(config.economy.initial_reference_price[target.value]))
    assert shock.confiscated == {first: (4, price * 4), second: (2, price * 2)}
    assert state.account(first).quantity(target) == 1
    # Its one free unit was not enough: the order for 3 is cut to the 2 left.
    assert state.account(second).quantity(target) == 2
    assert state.account(second).reserved(target) == 2
    assert shock.orders_cut == {second: (3, 2)}
    (order,) = engine.market.open_orders_of(second)
    assert order.remaining == 2
    assert shock.details["held_removed"] == 6


def test_confiscation_takes_from_adventurers_before_artisans(tmp_path: Path) -> None:
    """Adventurers sell the stock cheap; what is left stays with those who craft."""
    config = make_config(
        run={"scenario": Scenario.RESOURCE_SCARCITY.value},
        phases={"base_ticks": 10, "shock_ticks": 5, "recovery_ticks": 10},
        time={"horizon_ticks": 25},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
        shocks={"resource_scarcity": {"held_keep": 3}},
    )
    target = Resource(config.shocks.resource_scarcity.target_resource)
    state = build_world(config)
    for agent_id in state.agent_ids:  # some start with a little of it
        account = state.account(agent_id)
        account.consume(target, account.quantity(target))
    adventurer = state.agents_with_role(Role.ADVENTURER)[0]
    artisan = state.agents_with_role(Role.ARTISAN)[0]
    state.account(adventurer).receive(target, 2)
    state.account(artisan).receive(target, 5)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        shock = engine.shocks.apply(config.phases.shock_tick)

    # Seven held, three kept: the adventurer's two go first, though it held fewer.
    price = Decimal(str(config.economy.initial_reference_price[target.value]))
    assert shock.confiscated == {adventurer: (2, price * 2), artisan: (2, price * 2)}
    assert state.account(adventurer).quantity(target) == 0
    assert state.account(artisan).quantity(target) == 3


def test_confiscating_all_an_adventurer_offers_cancels_its_order_and_pays_it(
    tmp_path: Path,
) -> None:
    """Everything it held was on a sell order: the order goes, the gold comes."""
    config = make_config(
        run={"scenario": Scenario.RESOURCE_SCARCITY.value},
        phases={"base_ticks": 10, "shock_ticks": 5, "recovery_ticks": 10},
        time={"horizon_ticks": 25},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
        shocks={"resource_scarcity": {"held_keep": 0}},
    )
    target = Resource(config.shocks.resource_scarcity.target_resource)
    state = build_world(config)
    for agent_id in state.agent_ids:  # some start with a little of it
        account = state.account(agent_id)
        account.consume(target, account.quantity(target))
    seller = state.agents_with_role(Role.ADVENTURER)[0]
    account = state.account(seller)
    account.receive(target, 3)
    path = tmp_path / "run.jsonl"
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        order = engine.market.place(
            agent_id=seller, side=OrderSide.SELL, good=target, quantity=3, price=Decimal("9"), tick=0
        )
        gold_before = account.gold
        shock = engine.shocks.apply(config.phases.shock_tick)

    price = Decimal(str(config.economy.initial_reference_price[target.value]))
    assert account.quantity(target) == 0
    assert account.reserved(target) == 0
    assert account.gold == gold_before + price * 3
    assert order.status.value == "cancelled"
    assert engine.market.open_orders_of(seller) == []
    assert shock.confiscated == {seller: (3, price * 3)}
    assert shock.orders_cut == {seller: (3, 0)}
    (line,) = confiscation_lines(engine, account)  # told while the shock lasts
    assert f"took 3 {target.value}" in line
    assert f"paid you {(price * 3).quantize(Decimal('0.01'))} gold" in line
    assert "your sell orders for 3 were withdrawn" in line


def test_confiscated_holders_are_paid_and_told(tmp_path: Path) -> None:
    """The government buys the held stock out at its guide value, logged per agent."""
    config = make_config(
        run={"scenario": Scenario.RESOURCE_SCARCITY.value},
        phases={"base_ticks": 10, "shock_ticks": 5, "recovery_ticks": 10},
        time={"horizon_ticks": 25},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
        shocks={"resource_scarcity": {"held_keep": 0}},
    )
    target = Resource(config.shocks.resource_scarcity.target_resource)
    state = build_world(config)
    holder = state.agent_ids[0]
    state.account(holder).receive(target, 4)
    path = tmp_path / "run.jsonl"
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        while engine.tick <= config.phases.shock_tick:
            engine.run_cycle()

    events = [e for e in read_events(path) if e.type is EventType.GOODS_CONFISCATED]
    assert events and all(e.tick == config.phases.shock_tick for e in events)
    taken = {e.agent_id: e.payload for e in events}
    assert taken[holder]["good"] == target.value
    price = Decimal(str(config.economy.initial_reference_price[target.value]))
    assert all(Decimal(p["gold"]) == price * p["quantity"] for p in taken.values())

    line = next(s for s in agent_status(config, engine, state.account(holder)) if "Confiscated" in s)
    assert f"the government took {taken[holder]['quantity']} {target.value}" in line


def test_appearance_recovers_after_reversion(tmp_path: Path) -> None:
    config = short_config(Scenario.RESOURCE_SCARCITY)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=20)

    assert engine.shocks.active is None
    assert engine.zones.spawn_multiplier(IRON) == Decimal(1)


def test_scarcity_actually_depletes_iron_when_drawn_upon(tmp_path: Path) -> None:
    """The shock must bite while in force, not merely be recorded."""
    config = short_config(Scenario.RESOURCE_SCARCITY)
    # The configured target: measuring iron under a herb shock compared two
    # noisy samples of an unchanged rate.
    target = Resource(config.shocks.resource_scarcity.target_resource)
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)

        taken: list[int] = []

        def drain(e: Engine) -> None:
            # Take everything that appeared, so each cycle measures the inflow.
            taken.append(sum(zone.take(target, zone.stock(target)) for zone in e.zones.zones()))

        for _ in range(config.phases.shock_tick):  # base phase, normal appearance
            engine.run_cycle(drain)
        normal = sum(taken[-5:])

        taken.clear()
        for _ in range(5):  # shock phase, reduced appearance
            engine.run_cycle(drain)
        scarce = sum(taken)

    assert scarce < normal
