"""Macro-cycle loop, and the determinism gate for phase 1.

The determinism test is the binding one (IMPLEMENTATION.md sections 4.1 and 6): the same
seed with the same decisions must produce byte-identical logs. It runs over a
workload that genuinely exercises the order book -- crossing and resting orders,
partial fills, sweeps, expiries and cancellations -- because a determinism test
over an idle economy proves nothing.
"""

from __future__ import annotations

import contextlib
from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.agents import RuleBasedDecisionPhase
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import ALL_GOODS, OrderSide, Product, Resource, Scenario
from rpg_economy.engine import Engine, OrderRejected, WorldState, build_world
from rpg_economy.engine.rng import STREAM_TEST
from rpg_economy.engine.state import ZERO
from rpg_economy.logging_ import EventLogWriter, EventType, log_digest, read_events
from tests.conftest import gold_flow, make_config, make_metadata

IRON = Resource.IRON

#: The configured horizon, so tick-arithmetic tests follow the configured phases
#: instead of pinning a number that calibration will move.
HORIZON = make_config().time.horizon_ticks

#: Endowment that lets every agent both buy and sell from tick zero.
STOCKED = {
    "economy": {
        "initial_inventory": {
            role: {good.value: 40 for good in ALL_GOODS}
            for role in ("adventurer", "smithing", "woodworking", "alchemy")
        }
    }
}


class RandomTrading:
    """A decision phase that submits pseudo-random orders.

    Stands in for the agent layer. It draws from the engine's own seeded stream,
    so a run is fully determined by the run seed -- which is exactly the property
    under test.
    """

    def __init__(self, orders_per_cycle: int = 6) -> None:
        self._orders_per_cycle = orders_per_cycle

    def __call__(self, engine: Engine) -> None:
        rng = engine.random.stream(STREAM_TEST)
        agents = engine.state.agent_ids
        goods = list(ALL_GOODS[:4])

        for _ in range(self._orders_per_cycle):
            agent = rng.choice(agents)
            side = OrderSide.BUY if rng.random() < 0.5 else OrderSide.SELL
            good = rng.choice(goods)
            quantity = rng.randint(1, 5)
            price = Decimal(rng.randint(70, 130)) / 10

            # An agent that cannot afford or cover the order simply does not
            # trade. Rejections are part of the workload, not a failure.
            with contextlib.suppress(OrderRejected):
                engine.submit_order(agent, side, good, quantity, price)

            # Occasionally withdraw liquidity, exercising the cancel path.
            if rng.random() < 0.15:
                resting = engine.market.open_orders_of(agent)
                if resting:
                    engine.market.cancel(rng.choice(resting).order_id)


def run_to_log(path: Path, config: SimulationConfig, cycles: int = 40) -> None:
    """Execute a workload and write its log to ``path``."""
    state = build_world(config)
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        trading = RandomTrading()
        for _ in range(cycles):
            engine.run_cycle(trading)


@pytest.fixture
def stocked_config() -> SimulationConfig:
    return make_config(**STOCKED)


# ---------------------------------------------------------------------------
# Cycle structure (design v1.0 s8.1)
# ---------------------------------------------------------------------------
def test_default_config_makes_a_tick_a_macro_cycle(config: SimulationConfig) -> None:
    assert config.time.ticks_per_cycle == 1
    assert config.time.total_cycles == config.time.horizon_ticks


def test_a_cycle_advances_exactly_ticks_per_cycle(tmp_path: Path) -> None:
    config = make_config(time={"ticks_per_cycle": 4, "horizon_ticks": HORIZON})
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        assert engine.tick == 0

        engine.run_cycle()
        assert engine.tick == 4
        assert engine.cycle == 1

        engine.run_cycle()
        assert engine.tick == 8


def test_the_decision_phase_runs_at_the_cycles_last_tick(tmp_path: Path) -> None:
    """Orders must be stamped inside the cycle that placed them."""
    config = make_config(time={"ticks_per_cycle": 4, "horizon_ticks": HORIZON}, **STOCKED)
    seen: list[int] = []

    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run_cycle(lambda e: seen.append(e.tick))
        engine.run_cycle(lambda e: seen.append(e.tick))

    assert seen == [3, 7]


def test_mechanical_phase_precedes_the_decision_phase(tmp_path: Path) -> None:
    """All of a cycle's ticks resolve before any activation acts on the world."""
    config = make_config(time={"ticks_per_cycle": 5, "horizon_ticks": HORIZON}, **STOCKED)
    state = build_world(config)
    path = tmp_path / "run.jsonl"
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run_cycle(lambda e: e.submit_order("woodworking_1", OrderSide.SELL, IRON, 1, "10"))

    events = list(read_events(path))
    tick_seqs = [e.seq for e in events if e.type is EventType.TICK]
    placed_seq = next(e.seq for e in events if e.type is EventType.ORDER_PLACED)

    assert len(tick_seqs) == 5
    assert max(tick_seqs) < placed_seq


def test_run_reaches_the_planned_horizon(tmp_path: Path) -> None:
    config = make_config(time={"ticks_per_cycle": 5, "horizon_ticks": HORIZON})
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run()

    assert engine.tick == config.time.horizon_ticks
    assert engine.cycle == config.time.total_cycles


def test_a_partial_trailing_cycle_is_rejected() -> None:
    """Its agents would get fewer activation opportunities than everyone else's."""
    from rpg_economy.config import ConfigError

    with pytest.raises(ConfigError, match="whole number of"):
        make_config(time={"ticks_per_cycle": 7, "horizon_ticks": HORIZON + 1})


def test_order_expiry_happens_in_the_mechanical_phase(tmp_path: Path) -> None:
    config = make_config(market={"order_expiry_ticks": 2}, **STOCKED)
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run_cycle(lambda e: e.submit_order("woodworking_1", OrderSide.SELL, IRON, 1, "10"))
        order = engine.market.order(1)
        assert order.is_resting

        engine.run_cycle()
        assert order.is_resting  # tick 1: not yet due

        engine.run_cycle()
        assert not order.is_resting  # tick 2: expired


# ---------------------------------------------------------------------------
# The determinism gate
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_same_seed_produces_an_identical_log(
    tmp_path: Path, stocked_config: SimulationConfig
) -> None:
    """The binding invariant: same seed and same decisions, identical run."""
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_to_log(first, stocked_config)
    run_to_log(second, stocked_config)

    assert log_digest(first) == log_digest(second)


@pytest.mark.invariant
def test_the_determinism_workload_actually_exercises_the_book(
    tmp_path: Path, stocked_config: SimulationConfig
) -> None:
    """Guards the gate: determinism over an idle economy would prove nothing."""
    path = tmp_path / "run.jsonl"
    run_to_log(path, stocked_config)

    counts: dict[EventType, int] = {}
    for event in read_events(path):
        counts[event.type] = counts.get(event.type, 0) + 1

    for required in (
        EventType.ORDER_PLACED,
        EventType.TRADE_EXECUTED,
        EventType.ORDER_FILLED,
        EventType.ORDER_PARTIALLY_FILLED,
        EventType.ORDER_EXPIRED,
        EventType.ORDER_CANCELLED,
    ):
        assert counts.get(required, 0) > 0, f"workload never produced {required}"


def test_a_different_seed_produces_a_different_log(
    tmp_path: Path, stocked_config: SimulationConfig
) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_to_log(first, stocked_config)
    run_to_log(second, make_config(**STOCKED, run={"seed": 999}))

    assert log_digest(first) != log_digest(second)


def test_repeated_runs_agree_on_final_state(
    tmp_path: Path, stocked_config: SimulationConfig
) -> None:
    def final_holdings(label: str) -> dict[str, tuple[Decimal, int]]:
        state = build_world(stocked_config)
        with EventLogWriter(tmp_path / f"{label}.jsonl", make_metadata(stocked_config)) as log:
            engine = Engine(stocked_config, state, log)
            trading = RandomTrading()
            for _ in range(30):
                engine.run_cycle(trading)
        return {
            a: (state.account(a).gold, state.account(a).quantity(IRON)) for a in state.agent_ids
        }

    assert final_holdings("first") == final_holdings("second")


# ---------------------------------------------------------------------------
# Conservation over a full workload
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_long_run_conserves_gold_and_goods(
    tmp_path: Path, stocked_config: SimulationConfig
) -> None:
    state = build_world(stocked_config)
    gold_before = state.total_gold()
    iron_before = state.total_quantity(IRON)

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(stocked_config)) as log:
        engine = Engine(stocked_config, state, log)
        trading = RandomTrading()
        for _ in range(60):
            engine.run_cycle(trading)

    assert state.total_gold() == gold_before + gold_flow(tmp_path / "run.jsonl")
    assert state.total_quantity(IRON) == iron_before


@pytest.mark.invariant
def test_no_account_is_ever_overdrawn_or_short(
    tmp_path: Path, stocked_config: SimulationConfig
) -> None:
    state = build_world(stocked_config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(stocked_config)) as log:
        engine = Engine(stocked_config, state, log)
        trading = RandomTrading()
        for _ in range(60):
            engine.run_cycle(trading)
            for agent_id in state.agent_ids:
                account = state.account(agent_id)
                assert account.gold >= ZERO
                assert account.available_gold >= ZERO
                assert account.reserved_gold >= ZERO
                for good in ALL_GOODS:
                    assert account.quantity(good) >= 0
                    assert account.available_quantity(good) >= 0
                    assert account.reserved(good) >= 0


@pytest.mark.invariant
def test_reservations_are_fully_released_once_the_book_empties(
    tmp_path: Path, stocked_config: SimulationConfig
) -> None:
    """Every resting order must eventually expire and give its holdings back."""
    state = build_world(stocked_config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(stocked_config)) as log:
        engine = Engine(stocked_config, state, log)
        trading = RandomTrading()
        for _ in range(20):
            engine.run_cycle(trading)
        # Let every outstanding order age past its expiry with no new orders.
        for _ in range(stocked_config.market.order_expiry_ticks + 2):
            engine.run_cycle()

    for agent_id in state.agent_ids:
        account = state.account(agent_id)
        assert account.reserved_gold == ZERO
        assert all(units == 0 for units in account.reserved_inventory.values())


# ---------------------------------------------------------------------------
# Market logging (design v1.0 s5.2)
# ---------------------------------------------------------------------------
def test_every_trade_is_logged_with_counterparties_and_roles(tmp_path: Path) -> None:
    config = make_config(**STOCKED)
    path = tmp_path / "run.jsonl"
    state = build_world(config)
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run_cycle(
            lambda e: (
                e.submit_order("woodworking_1", OrderSide.SELL, IRON, 2, "10"),
                e.submit_order("woodworking_2", OrderSide.BUY, IRON, 2, "10"),
            )
        )

    trades = [e for e in read_events(path) if e.type is EventType.TRADE_EXECUTED]
    assert len(trades) == 1
    payload = trades[0].payload
    assert payload["buyer_id"] == "woodworking_2"
    assert payload["seller_id"] == "woodworking_1"
    assert payload["buyer_role"] == "artisan"
    assert payload["seller_role"] == "artisan"
    assert payload["quantity"] == 2
    assert payload["price"] == "10"
    assert payload["value"] == "20"
    assert trades[0].tick == 0


def test_order_lifecycle_transitions_are_all_logged(tmp_path: Path) -> None:
    """Placement, partial fill, fill, cancellation and expiry each leave a record."""
    config = make_config(market={"order_expiry_ticks": 2}, **STOCKED)
    path = tmp_path / "run.jsonl"
    state = build_world(config)

    def place_and_partially_fill(engine: Engine) -> None:
        engine.submit_order("woodworking_1", OrderSide.SELL, IRON, 5, "10")
        engine.submit_order("woodworking_2", OrderSide.BUY, IRON, 2, "10")

    def cancel_and_rest(engine: Engine) -> None:
        engine.market.cancel(engine.market.open_orders_of("woodworking_1")[0].order_id)
        engine.submit_order("woodworking_1", OrderSide.SELL, IRON, 1, "99")

    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run_cycle(place_and_partially_fill)
        engine.run_cycle(cancel_and_rest)
        for _ in range(3):
            engine.run_cycle()

    types = {e.type for e in read_events(path)}
    assert {
        EventType.ORDER_PLACED,
        EventType.ORDER_PARTIALLY_FILLED,
        EventType.ORDER_FILLED,
        EventType.ORDER_CANCELLED,
        EventType.ORDER_EXPIRED,
        EventType.TRADE_EXECUTED,
    } <= types


def test_logged_prices_are_exact_strings(tmp_path: Path) -> None:
    """Decimal money must survive into the log without a float round-trip."""
    config = make_config(**STOCKED)
    path = tmp_path / "run.jsonl"
    state = build_world(config)
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run_cycle(lambda e: e.submit_order("woodworking_1", OrderSide.SELL, IRON, 1, "0.10"))

    placed = [e for e in read_events(path) if e.type is EventType.ORDER_PLACED]
    assert placed[0].payload["limit_price"] == "0.10"


# ---------------------------------------------------------------------------
# The full-run determinism gate (phase 1 part 2)
# ---------------------------------------------------------------------------
def run_full_scenario(path: Path, config: SimulationConfig) -> WorldState:
    """A complete run of the primary scenario, driven by rule-based agents."""
    state = build_world(config)
    with EventLogWriter(path, make_metadata(config)) as log:
        Engine(config, state, log).run(RuleBasedDecisionPhase(config))
    return state


@pytest.mark.invariant
def test_a_full_resource_scarcity_run_is_deterministic(tmp_path: Path) -> None:
    """The phase 1 gate: a whole run of the primary scenario, twice, identically.

    This exercises every mechanic together -- expeditions resolving against
    per-agent streams, production, durability, consumption, the order book, and
    the shock -- which is where any hidden non-determinism (dict ordering, a
    shared generator, a float comparison) would surface.
    """
    config = make_config(run={"scenario": Scenario.RESOURCE_SCARCITY.value})
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_full_scenario(first, config)
    run_full_scenario(second, config)

    assert log_digest(first) == log_digest(second)


@pytest.mark.invariant
def test_the_full_run_exercises_every_mechanic(tmp_path: Path) -> None:
    """Guards the gate: determinism over a run that does nothing proves nothing."""
    config = make_config(run={"scenario": Scenario.RESOURCE_SCARCITY.value})
    path = tmp_path / "run.jsonl"
    run_full_scenario(path, config)

    counts: dict[EventType, int] = {}
    for event in read_events(path):
        counts[event.type] = counts.get(event.type, 0) + 1

    for required in (
        EventType.EXPEDITION_STARTED,
        EventType.EXPEDITION_RESOLVED,
        EventType.PRODUCTION_COMPLETED,
        EventType.EQUIPMENT_EQUIPPED,
        EventType.EQUIPMENT_BROKEN,
        EventType.CONSUMPTION,
        EventType.TRADE_EXECUTED,
        EventType.ORDER_PLACED,
        EventType.ORDER_EXPIRED,
        EventType.SHOCK_APPLIED,
        EventType.SHOCK_REVERTED,
    ):
        assert counts.get(required, 0) > 0, f"the run never produced {required}"


def test_the_shock_fires_at_its_configured_tick_in_a_full_run(tmp_path: Path) -> None:
    config = make_config(run={"scenario": Scenario.RESOURCE_SCARCITY.value})
    path = tmp_path / "run.jsonl"
    run_full_scenario(path, config)

    shocks = [
        (e.type, e.tick, e.payload.get("target_resource"))
        for e in read_events(path)
        if e.type in {EventType.SHOCK_APPLIED, EventType.SHOCK_REVERTED}
    ]
    assert shocks == [
        (EventType.SHOCK_APPLIED, config.phases.shock_tick, "herbs"),
        (EventType.SHOCK_REVERTED, config.phases.reversion_tick, "herbs"),
    ]


def test_a_different_seed_changes_a_full_run(tmp_path: Path) -> None:
    config = make_config(run={"scenario": Scenario.RESOURCE_SCARCITY.value})
    other = make_config(run={"scenario": Scenario.RESOURCE_SCARCITY.value, "seed": 4242})
    run_full_scenario(tmp_path / "a.jsonl", config)
    run_full_scenario(tmp_path / "b.jsonl", other)

    assert log_digest(tmp_path / "a.jsonl") != log_digest(tmp_path / "b.jsonl")


@pytest.mark.invariant
def test_a_full_run_conserves_gold(tmp_path: Path) -> None:
    """Gold enters only as expedition loot and leaves only as taxes and storage."""
    config = make_config(run={"scenario": Scenario.RESOURCE_SCARCITY.value})
    expected = build_world(config).total_gold()
    state = run_full_scenario(tmp_path / "run.jsonl", config)

    assert state.total_gold() == expected + gold_flow(tmp_path / "run.jsonl")


@pytest.mark.invariant
def test_a_full_run_leaves_no_account_overdrawn(tmp_path: Path) -> None:
    config = make_config(run={"scenario": Scenario.RESOURCE_SCARCITY.value})
    state = run_full_scenario(tmp_path / "run.jsonl", config)

    for agent_id in state.agent_ids:
        account = state.account(agent_id)
        assert account.gold >= ZERO
        assert account.available_gold >= ZERO
        for good in ALL_GOODS:
            assert account.quantity(good) >= 0
            assert account.available_quantity(good) >= 0


def test_every_scenario_runs_to_the_horizon(tmp_path: Path) -> None:
    """All four implementable shock types survive a complete run."""
    for scenario in (
        Scenario.RESOURCE_SCARCITY,
        Scenario.DEMAND_SURGE,
        Scenario.PRODUCTIVITY_DROP,
        Scenario.LIQUIDITY_LOSS,
    ):
        config = make_config(run={"scenario": scenario.value})
        state = run_full_scenario(tmp_path / f"{scenario.value}.jsonl", config)
        assert state.tick == config.time.horizon_ticks


# ---------------------------------------------------------------------------
# Condition B end to end (phase 2)
# ---------------------------------------------------------------------------
def run_condition_b(path: Path, config: SimulationConfig) -> WorldState:
    """A full run driven by triggers and the neutral scheduler."""
    from rpg_economy.agents import execute, policy_for
    from rpg_economy.orchestration import DecentralizedOrchestrator

    state = build_world(config)
    policies: dict = {}

    def activate(engine: Engine, agent_id: str) -> None:
        account = engine.state.account(agent_id)
        policy = policies.setdefault(account.role, policy_for(account.role, config))
        action = policy.decide(engine, account, engine.random.agent(agent_id))
        execute(engine, agent_id, action)

    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run(DecentralizedOrchestrator(config, engine, activate))
    return state


@pytest.mark.invariant
def test_condition_b_is_deterministic(tmp_path: Path) -> None:
    """Triggers and scheduling must not introduce non-determinism."""
    config = make_config(run={"condition": "B"})
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_condition_b(first, config)
    run_condition_b(second, config)

    assert log_digest(first) == log_digest(second)


def test_condition_b_exercises_the_activation_machinery(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    path = tmp_path / "run.jsonl"
    run_condition_b(path, config)

    counts: dict[EventType, int] = {}
    for event in read_events(path):
        counts[event.type] = counts.get(event.type, 0) + 1

    for required in (
        EventType.TRIGGER_FIRED,
        EventType.ACTIVATION_REQUESTED,
        EventType.ACTIVATION_SERVED,
        EventType.ACTIVATION_DEFERRED,
        EventType.AGENT_OUTPUT,
        EventType.TRADE_EXECUTED,
        EventType.SHOCK_APPLIED,
    ):
        assert counts.get(required, 0) > 0, f"condition B never produced {required}"


@pytest.mark.invariant
def test_condition_b_respects_the_k_budget_every_cycle(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    path = tmp_path / "run.jsonl"
    run_condition_b(path, config)

    for event in read_events(path):
        if event.type is EventType.CYCLE_END:
            assert event.payload["effective_activations"] <= config.time.k_activations


@pytest.mark.invariant
def test_condition_b_conserves_gold(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    expected = build_world(config).total_gold()
    state = run_condition_b(tmp_path / "run.jsonl", config)
    assert state.total_gold() == expected + gold_flow(tmp_path / "run.jsonl")


def test_condition_b_and_a_seed_trio_share_the_same_shock(tmp_path: Path) -> None:
    """A/B/C must receive the same shock at the same tick for a given seed."""
    shocks = []
    for condition in ("A", "B"):
        config = make_config(run={"condition": condition})
        path = tmp_path / f"{condition}.jsonl"
        if condition == "B":
            run_condition_b(path, config)
        else:
            run_full_scenario(path, config)
        shocks.append(
            [
                (e.type, e.tick)
                for e in read_events(path)
                if e.type in {EventType.SHOCK_APPLIED, EventType.SHOCK_REVERTED}
            ]
        )
    assert shocks[0] == shocks[1]


# ---------------------------------------------------------------------------
# Zone stock record (design v1.0 s15)
# ---------------------------------------------------------------------------
class SendIdleAdventurers:
    """A decision phase that keeps every adventurer on expeditions, rotating zones."""

    def __call__(self, engine: Engine) -> None:
        from rpg_economy.engine import ExpeditionRejected

        zones = engine.zones.zone_ids
        for index, agent_id in enumerate(engine.state.agent_ids):
            if not agent_id.startswith(("warrior", "mage", "ranger")):
                continue
            with contextlib.suppress(ExpeditionRejected):
                engine.start_expedition(agent_id, zones[(engine.tick + index) % len(zones)])


def test_the_zone_stock_is_logged_and_reconciles(tmp_path: Path) -> None:
    """Each tick's stock is the previous one, minus expedition takes, plus appearance."""
    config = make_config()
    path = tmp_path / "run.jsonl"
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        for _ in range(60):
            engine.run_cycle(SendIdleAdventurers())

    events = list(read_events(path))
    spawns = {e.tick: e.payload for e in events if e.type is EventType.RESOURCES_SPAWNED}
    assert sorted(spawns) == list(range(60))

    taken: dict[int, dict[str, dict[str, int]]] = {}
    for event in events:
        if event.type is EventType.EXPEDITION_RESOLVED:
            zone = taken.setdefault(event.tick, {}).setdefault(event.payload["zone_id"], {})
            for resource, units in event.payload["yielded"].items():
                zone[resource] = zone.get(resource, 0) + units
    assert taken, "the workload should have resolved some expeditions"

    for tick in range(1, 60):
        before, now = spawns[tick - 1]["stock"], spawns[tick]["stock"]
        for zone, stock in now.items():
            for resource, units in stock.items():
                expected = (
                    before[zone][resource]
                    - taken.get(tick - 1, {}).get(zone, {}).get(resource, 0)
                    + spawns[tick]["spawned"].get(zone, {}).get(resource, 0)
                )
                assert units == expected, (tick, zone, resource)


def test_adventurers_do_not_burn_potions_while_idle(tmp_path) -> None:
    """Potions are used on expeditions, not by the passage of time."""
    from rpg_economy.domain import Product
    from rpg_economy.engine import Engine
    from rpg_economy.logging_ import EventLogWriter
    from tests.conftest import make_metadata

    config = make_config()
    state = build_world(config)
    before = state.account("warrior_1").quantity(Product.HEALTH_POTION)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        for _ in range(20):
            engine.run_cycle(None)
    assert state.account("warrior_1").quantity(Product.HEALTH_POTION) == before


# ---------------------------------------------------------------------------
# Gear wears on expeditions only
# ---------------------------------------------------------------------------
def test_equipment_does_not_wear_in_town(tmp_path: Path) -> None:
    from rpg_economy.domain import Product

    config = make_config()
    state = build_world(config)
    state.account("warrior_1").equipment[Product.MELEE_WEAPON] = 3
    state.account("warrior_1").equipment[Product.ARMOR] = 1
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        for _ in range(50):
            engine.run_cycle(None)

    assert state.account("warrior_1").equipment == {Product.MELEE_WEAPON: 3, Product.ARMOR: 1}
    assert not any(
        e.type is EventType.EQUIPMENT_BROKEN for e in read_events(tmp_path / "run.jsonl")
    )


def test_an_expedition_wears_equipment_by_the_configured_loss(tmp_path: Path) -> None:
    from rpg_economy.domain import Product

    config = make_config()
    loss = config.expeditions.durability_loss_per_expedition
    state = build_world(config)
    state.account("warrior_1").equipment[Product.MELEE_WEAPON] = 10
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.start_expedition("warrior_1", "green_hollow")
        while state.account("warrior_1").expedition is not None:
            engine.run_cycle(None)
        for _ in range(10):
            engine.run_cycle(None)

    assert state.account("warrior_1").durability_of(Product.MELEE_WEAPON) == 10 - loss


# ---------------------------------------------------------------------------
# Gold sources and sinks: expedition loot, storage and taxes
# ---------------------------------------------------------------------------
def test_expedition_loot_is_logged_with_the_resolution(tmp_path: Path) -> None:
    config = make_config()
    state = build_world(config)
    path = tmp_path / "run.jsonl"
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        before = state.account("warrior_1").gold
        engine.start_expedition("warrior_1", "green_hollow")
        while state.account("warrior_1").expedition is not None:
            engine.run_cycle(None)

    (resolved,) = [e for e in read_events(path) if e.type is EventType.EXPEDITION_RESOLVED]
    loot = Decimal(resolved.payload["gold"])
    from rpg_economy.domain import RiskLevel

    (started,) = [e for e in read_events(path) if e.type is EventType.EXPEDITION_STARTED]
    if resolved.payload["outcome"] == "success":
        # Each item carried (its weapon, its potion) adds its share of the gold.
        shares = sum(
            config.expeditions.gear_gold_bonus.get(Product(item), Decimal(0))
            for item in started.payload["gear_bonus"]
        )
        base = config.expeditions.gold_loot[RiskLevel(resolved.payload["risk"])]
        assert loot == base * (1 + shares)
    else:
        assert loot == 0
    assert state.account("warrior_1").gold == before + loot


#: No taxes, so the storage tests see storage alone.
NO_TAXES = {"profit_rate": 0, "wealth_rate": 0}


def _charged(tmp_path: Path, config: SimulationConfig, cycles: int, prepare=None, kind=None):
    """Run ``cycles`` idle cycles; return the state and the storage or tax payments."""
    state = build_world(config)
    if prepare is not None:
        prepare(state)
    path = tmp_path / "run.jsonl"
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        for _ in range(cycles):
            engine.run_cycle(None)
    wanted = EventType.STORAGE_PAID if kind is None else EventType.TAX_PAID
    paid = [
        e
        for e in read_events(path)
        if e.type is wanted and (kind is None or e.payload["tax"] == kind)
    ]
    return state, paid


def test_artisans_pay_storage_per_unit_held(tmp_path: Path) -> None:
    config = make_config(
        economy={
            "taxes": NO_TAXES,
            "storage_cost_per_unit": {"artisan": "0.5", "adventurer": "0"},
            "charge_floor": 0,
            "floor_at_initial_gold": False,
        }
    )
    _, paid = _charged(tmp_path, config, cycles=1)
    by_agent = {e.agent_id: e for e in paid}
    initial = build_world(config)
    for agent_id in ("smithing_1", "woodworking_1"):
        units = sum(initial.account(agent_id).inventory.values())
        assert by_agent[agent_id].payload["units"] == units
        assert Decimal(by_agent[agent_id].payload["amount"]) == Decimal("0.5") * units
    # Adventurers pay no storage.
    assert not any(e.agent_id.startswith(("warrior", "mage", "ranger")) for e in paid)


@pytest.mark.invariant
def test_storage_is_paid_only_from_unreserved_gold_and_never_below_zero(tmp_path: Path) -> None:
    config = make_config(
        economy={
            "taxes": NO_TAXES,
            "storage_cost_per_unit": {"artisan": 100},
            "charge_floor": 0,
            "floor_at_initial_gold": False,
        }
    )

    def prepare(state: WorldState) -> None:
        smith = state.account("smithing_1")
        smith.gold = Decimal("7")
        smith.reserve_gold(Decimal("4"))  # backs a resting bid: not spendable
        state.account("woodworking_1").gold = Decimal("0")

    state, paid = _charged(tmp_path, config, cycles=2, prepare=prepare)
    smith = state.account("smithing_1")
    charged = [Decimal(e.payload["amount"]) for e in paid if e.agent_id == "smithing_1"]

    assert charged == [Decimal("3")]
    assert smith.gold == Decimal("4")
    assert smith.available_gold == ZERO
    assert state.account("woodworking_1").gold == ZERO
    assert not any(e.agent_id == "woodworking_1" for e in paid)


def test_storage_never_takes_anyone_below_the_floor(tmp_path: Path) -> None:
    config = make_config(
        economy={
            "taxes": NO_TAXES,
            "storage_cost_per_unit": {"artisan": 100},
            "charge_floor": 60,
            "floor_at_initial_gold": False,
        }
    )

    def prepare(state: WorldState) -> None:
        state.account("smithing_1").gold = Decimal("68")
        state.account("woodworking_1").gold = Decimal("70")
        state.account("alchemy_1").gold = Decimal("50")

    state, paid = _charged(tmp_path, config, cycles=3, prepare=prepare)

    assert state.account("smithing_1").gold == Decimal("60")
    assert state.account("woodworking_1").gold == Decimal("60")
    assert state.account("alchemy_1").gold == Decimal("50")
    assert not any(e.agent_id == "alchemy_1" for e in paid)
