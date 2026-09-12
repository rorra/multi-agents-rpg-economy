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

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import ALL_GOODS, OrderSide, Resource
from rpg_economy.engine import Engine, OrderRejected, build_world
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
