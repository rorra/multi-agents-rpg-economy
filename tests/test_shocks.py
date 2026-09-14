"""Shocks: finite duration, exact timing, and per-scenario mechanisms."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.agents import RuleBasedDecisionPhase
from rpg_economy.agents.observation import agent_status, confiscation_lines
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import (
    AdventurerSubclass,
    ArtisanSpecialization,
    OrderSide,
    Product,
    Resource,
    Role,
    Scenario,
)
from rpg_economy.engine import Engine, ShockNotImplemented, build_world
from rpg_economy.engine.state import ZERO
from rpg_economy.logging_ import EventLogWriter, EventType, read_events
from tests.conftest import gold_flow, make_config, make_metadata

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


# ---------------------------------------------------------------------------
# Other scenarios
# ---------------------------------------------------------------------------
def test_productivity_drop_reduces_the_targets_capacity(tmp_path: Path) -> None:
    config = short_config(Scenario.PRODUCTIVITY_DROP)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=12)

    smithing, alchemy = ArtisanSpecialization.SMITHING, ArtisanSpecialization.ALCHEMY
    rates = config.economy.production_capacity_per_cycle
    assert engine.production.rate_of(smithing) < rates[smithing]
    assert engine.production.rate_of(alchemy) == rates[alchemy]


def test_productivity_recovers_after_reversion(tmp_path: Path) -> None:
    config = short_config(Scenario.PRODUCTIVITY_DROP)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=20)
    assert engine.production.capacity_multiplier(ArtisanSpecialization.SMITHING) == Decimal(1)


def test_demand_surge_raises_consumption(tmp_path: Path) -> None:
    config = short_config(Scenario.DEMAND_SURGE)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=12)

    target = config.shocks.demand_surge.target_product
    assert engine.shocks.consumption_multiplier(target) > Decimal(1)
    assert engine.shocks.consumption_multiplier(Resource.IRON) == Decimal(1)


def test_liquidity_loss_freezes_gold_without_destroying_it(tmp_path: Path) -> None:
    """Reduced purchasing capacity, not confiscation: total wealth is unchanged."""
    config = short_config(Scenario.LIQUIDITY_LOSS)
    state = build_world(config)
    total_before = state.total_gold()

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        for _ in range(12):
            engine.run_cycle()

        frozen = [a for a in state.agent_ids if state.account(a).frozen_gold > ZERO]
        assert frozen, "no agent had gold frozen"
        for agent_id in frozen:
            account = state.account(agent_id)
            assert account.available_gold < account.gold

    # Taxes and storage are the only gold that leaves; the freeze destroys none.
    assert state.total_gold() == total_before + gold_flow(tmp_path / "run.jsonl")


def test_liquidity_is_restored_at_reversion(tmp_path: Path) -> None:
    config = short_config(Scenario.LIQUIDITY_LOSS)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=20)

    for agent_id in engine.state.agent_ids:
        assert engine.state.account(agent_id).frozen_gold == ZERO


def test_liquidity_loss_freezes_the_configured_agents_only(tmp_path: Path) -> None:
    """The base configuration is a general credit crunch; a subset is honoured too
    (see the explicit-list test below)."""
    config = short_config(Scenario.LIQUIDITY_LOSS)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=12)

    frozen = {a for a in engine.state.agent_ids if engine.state.account(a).frozen_gold > ZERO}
    assert frozen
    assert frozen <= set(config.shocks.liquidity_loss.affected_agents)


# ---------------------------------------------------------------------------
# Who is affected is specified, not inferred (design v1.0 s10)
# ---------------------------------------------------------------------------
def frozen_agents(engine: Engine) -> list[str]:
    return [a for a in engine.state.agent_ids if engine.state.account(a).frozen_gold > ZERO]


def test_an_explicit_agent_list_is_honoured(tmp_path: Path) -> None:
    config = make_config(
        run={"scenario": Scenario.LIQUIDITY_LOSS.value},
        phases={"base_ticks": 10, "shock_ticks": 5, "recovery_ticks": 10},
        time={"horizon_ticks": 25},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
        shocks={
            "liquidity_loss": {
                "frozen_gold_fraction": 0.6,
                "affected_agents": ["woodworking_2", "warrior_1"],
                "affected_by_role": None,
            }
        },
    )
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=12)
    assert sorted(frozen_agents(engine)) == ["warrior_1", "woodworking_2"]


def test_a_role_stratified_count_freezes_that_many_of_each_role(tmp_path: Path) -> None:
    """The alternative specification, supported for larger populations."""
    by_role = {"artisan": 2, "adventurer": 2}
    config = make_config(
        run={"scenario": Scenario.LIQUIDITY_LOSS.value},
        phases={"base_ticks": 10, "shock_ticks": 5, "recovery_ticks": 10},
        time={"horizon_ticks": 25},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
        shocks={
            "liquidity_loss": {
                "frozen_gold_fraction": 0.6,
                "affected_agents": None,
                "affected_by_role": by_role,
            }
        },
    )
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=12)

    affected = frozen_agents(engine)
    for role, count in by_role.items():
        actual = sum(1 for a in affected if engine.state.account(a).role.value == role)
        assert actual == count


def test_the_default_subset_spans_roles_and_subclasses(tmp_path: Path) -> None:
    """The configured default must be a deliberate mix, not an ordering artifact.

    Ids encode both role and subclass, so any rule that just takes the first N
    of something lands on a biased set: a sorted prefix hits whichever roles
    sort first, and a per-role count at pilot scale takes two mages and no
    warrior. The default names its four agents explicitly.
    """
    config = short_config(Scenario.LIQUIDITY_LOSS)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=12)

    affected = frozen_agents(engine)
    roles = {engine.state.account(a).role for a in affected}
    specializations = {
        engine.state.account(a).specialization
        for a in affected
        if engine.state.account(a).role is Role.ADVENTURER
    }

    assert roles == {Role.ADVENTURER, Role.ARTISAN}, f"subset {affected} covers only {roles}"
    assert len(specializations) > 1, f"adventurers in {affected} are all {specializations}"


def test_a_missing_specification_is_rejected() -> None:
    from rpg_economy.config import ConfigError

    with pytest.raises(ConfigError, match="exactly one"):
        make_config(
            shocks={
                "liquidity_loss": {
                    "frozen_gold_fraction": 0.6,
                    "affected_agents": None,
                    "affected_by_role": None,
                }
            }
        )


def test_a_double_specification_is_rejected() -> None:
    from rpg_economy.config import ConfigError

    with pytest.raises(ConfigError, match="exactly one"):
        make_config(
            shocks={
                "liquidity_loss": {
                    "frozen_gold_fraction": 0.6,
                    "affected_agents": ["woodworking_1"],
                    "affected_by_role": {"artisan": 1},
                }
            }
        )


def test_a_subset_larger_than_the_population_is_rejected() -> None:
    from rpg_economy.config import ConfigError

    with pytest.raises(ConfigError, match="population has"):
        make_config(
            shocks={
                "liquidity_loss": {
                    "frozen_gold_fraction": 0.6,
                    "affected_agents": None,
                    "affected_by_role": {"artisan": 99},
                }
            }
        )


@pytest.mark.parametrize("by_role", [{"artisan": -1}, {"artisan": 2, "adventurer": -1}])
def test_a_negative_role_count_is_rejected(by_role: dict) -> None:
    """A negative count is not a subset: it would slice agents off the end of the role."""
    from rpg_economy.config import ConfigError

    with pytest.raises(ConfigError, match="greater than or equal to 0"):
        make_config(
            shocks={
                "liquidity_loss": {
                    "frozen_gold_fraction": 0.6,
                    "affected_agents": None,
                    "affected_by_role": by_role,
                }
            }
        )


def test_an_unknown_agent_id_is_rejected_at_load() -> None:
    """A typo in the subset fails when the config loads, not at the shock tick."""
    from rpg_economy.config import ConfigError

    with pytest.raises(ConfigError, match="does not have"):
        make_config(
            shocks={
                "liquidity_loss": {
                    "frozen_gold_fraction": 0.6,
                    "affected_agents": ["warrior_1", "nobody_1"],
                    "affected_by_role": None,
                }
            },
        )


def test_an_id_past_the_population_count_is_rejected_at_load() -> None:
    """Ids are checked against the ids the population mints, not just their prefix."""
    from rpg_economy.config import ConfigError

    config = make_config()
    too_far = f"warrior_{config.population.adventurers[AdventurerSubclass.WARRIOR] + 1}"
    with pytest.raises(ConfigError, match=too_far):
        make_config(
            shocks={
                "liquidity_loss": {
                    "frozen_gold_fraction": 0.6,
                    "affected_agents": [too_far],
                    "affected_by_role": None,
                }
            },
        )


def test_every_id_the_population_mints_is_accepted() -> None:
    config = make_config()
    ids = [agent_id for agent_id, _, _ in config.population.members()]
    assert sorted(ids) == build_world(config).agent_ids
    make_config(
        shocks={
            "liquidity_loss": {
                "frozen_gold_fraction": 0.6,
                "affected_agents": ids,
                "affected_by_role": None,
            }
        },
    )


def test_the_affected_subset_is_identical_across_seeds(tmp_path: Path) -> None:
    """It must be the same subset in every condition of a seed-trio."""
    subsets = []
    for seed in (11, 22, 33):
        config = short_config(Scenario.LIQUIDITY_LOSS, seed=seed)
        engine = run_scenario(tmp_path / f"{seed}.jsonl", config, cycles=12)
        subsets.append(
            [a for a in engine.state.agent_ids if engine.state.account(a).frozen_gold > ZERO]
        )
    assert subsets[0] == subsets[1] == subsets[2]


def test_noisy_info_is_not_implemented(tmp_path: Path) -> None:
    """Admissible only if defined before final runs (design s10)."""
    config = short_config(Scenario.NOISY_INFO)
    with pytest.raises(ShockNotImplemented, match="noisy_info"):
        run_scenario(tmp_path / "run.jsonl", config, cycles=12)


# ---------------------------------------------------------------------------
# The Director cannot touch shocks
# ---------------------------------------------------------------------------
def test_shocks_fire_regardless_of_agent_behaviour(tmp_path: Path) -> None:
    """Design s10: the Director can neither prevent nor modify a shock."""
    config = short_config(Scenario.RESOURCE_SCARCITY)
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        phase = RuleBasedDecisionPhase(config)
        for _ in range(25):
            engine.run_cycle(phase)

    applied = [
        e.tick for e in read_events(tmp_path / "run.jsonl") if e.type is EventType.SHOCK_APPLIED
    ]
    assert applied == [config.phases.shock_tick]


def test_demand_surge_multiplies_the_potions_an_expedition_uses(tmp_path: Path) -> None:
    """The surge bites through expedition supplies: each trip takes more potions."""
    config = short_config(Scenario.DEMAND_SURGE)
    engine = run_scenario(tmp_path / "run.jsonl", config, cycles=config.phases.shock_tick + 1)
    account = engine.state.account("warrior_1")
    target = config.shocks.demand_surge.target_product
    needed = engine.expeditions.supplies_for(account, engine.shocks.consumption_multiplier)
    assert target in config.expeditions.gear[AdventurerSubclass.WARRIOR]
    assert needed[target] > 1


# ---------------------------------------------------------------------------
# Shock efficacy (calibration diagnostic)
# ---------------------------------------------------------------------------
def test_a_control_run_never_applies_the_shock(tmp_path: Path) -> None:
    config = short_config(Scenario.RESOURCE_SCARCITY)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        Engine(config, build_world(config), log, shock_enabled=False).run(
            RuleBasedDecisionPhase(config)
        )
    kinds = {e.type for e in read_events(tmp_path / "run.jsonl")}
    assert EventType.SHOCK_APPLIED not in kinds


@pytest.mark.invariant
def test_the_freeze_does_not_depend_on_a_bid_being_out_at_the_shock() -> None:
    """Two agents with equal gold end with equal frozen gold once a bid returns."""
    from rpg_economy.engine.state import AgentAccount

    idle = AgentAccount(
        agent_id="a", role=Role.ARTISAN, specialization="smithing", gold=Decimal(100)
    )
    bidding = AgentAccount(
        agent_id="b", role=Role.ARTISAN, specialization="smithing", gold=Decimal(100)
    )
    bidding.reserve_gold(Decimal(50))

    idle.freeze_gold(Decimal("0.6"))
    bidding.freeze_gold(Decimal("0.6"))
    bidding.release_gold(Decimal(50))  # the bid expires during the shock

    assert idle.frozen_gold == bidding.frozen_gold == Decimal(60)
    assert idle.available_gold == bidding.available_gold == Decimal(40)


@pytest.mark.invariant
def test_the_freeze_nets_arrears_the_same_whether_gold_is_free_or_in_a_bid() -> None:
    """Reserved gold that will go to arrears once released is not frozen.

    Gold 1000, arrears 100, fraction 0.95: 855 frozen, nothing owed and 45
    available at the end, whether the gold was free at the shock or reserved
    behind a bid that expires during it.
    """
    from rpg_economy.engine.state import AgentAccount

    def account(agent_id: str) -> AgentAccount:
        acc = AgentAccount(
            agent_id=agent_id, role=Role.ARTISAN, specialization="smithing", gold=Decimal(1000)
        )
        acc.tax_arrears = Decimal(100)
        return acc

    def collect(acc: AgentAccount) -> None:  # what the cycle-start collection does
        paid = min(acc.tax_arrears, acc.free_gold)
        acc.gold -= paid
        acc.tax_arrears -= paid

    idle, bidding = account("a"), account("b")
    bidding.reserved_gold = Decimal(1000)  # placed before the arrears were assessed

    idle.freeze_gold(Decimal("0.95"), tick=100)
    bidding.freeze_gold(Decimal("0.95"), tick=100)
    assert bidding.freezable_reserved == Decimal(900)
    bidding.release_gold(Decimal(1000), placed_tick=90)  # the bid expires during the shock
    collect(idle)
    collect(bidding)

    for acc in (idle, bidding):
        assert acc.frozen_gold == Decimal(855)
        assert acc.tax_arrears == ZERO
        assert acc.available_gold == Decimal(45)
        assert acc.gold == Decimal(900)


def test_arrears_free_gold_covers_leave_the_reservation_freezable_in_full() -> None:
    from rpg_economy.engine.state import AgentAccount

    account = AgentAccount(
        agent_id="b", role=Role.ARTISAN, specialization="smithing", gold=Decimal(1000)
    )
    account.reserve_gold(Decimal(500))
    account.tax_arrears = Decimal(100)
    account.freeze_gold(Decimal("0.5"), tick=10)
    assert account.freezable_reserved == Decimal(500)
    assert account.frozen_gold == Decimal(200)  # half of 500 free less 100 owed


def test_a_bid_committed_before_the_freeze_can_still_fill() -> None:
    from rpg_economy.engine.state import AgentAccount

    account = AgentAccount(
        agent_id="b", role=Role.ARTISAN, specialization="smithing", gold=Decimal(100)
    )
    account.reserve_gold(Decimal(50))
    account.freeze_gold(Decimal("0.6"))
    account.spend_reserved_gold(Decimal(50))

    assert account.gold == Decimal(50)
    assert account.frozen_gold == Decimal(30)
    assert account.available_gold == Decimal(20)

    account.unfreeze_all_gold()
    account.release_gold(Decimal(0))
    assert account.available_gold == Decimal(50)


def test_noisy_info_is_refused_before_the_run_starts(tmp_path: Path) -> None:
    """No tick, and so no LLM call, happens for a scenario that cannot run."""
    config = short_config(Scenario.NOISY_INFO)
    with (
        EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log,
        pytest.raises(ShockNotImplemented),
    ):
        Engine(config, build_world(config), log)


# ---------------------------------------------------------------------------
# Which reserved gold the freeze reaches (design v1.0 s10, liquidity_loss)
# ---------------------------------------------------------------------------
FREEZE_SHOCK_TICK = 10
FREEZE_REVERSION_TICK = 30


def _freeze_config(**overrides) -> SimulationConfig:
    """warrior_1 alone, half its gold, and orders that expire inside the shock."""
    return make_config(
        **overrides,
        run={"scenario": Scenario.LIQUIDITY_LOSS.value},
        phases={"base_ticks": 10, "shock_ticks": 20, "recovery_ticks": 10},
        time={"horizon_ticks": 40},
        market={"order_expiry_ticks": 3},
        metrics={"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
        shocks={
            "liquidity_loss": {
                "frozen_gold_fraction": 0.5,
                "affected_agents": ["warrior_1"],
                "affected_by_role": None,
            }
        },
    )


def _run_with_bids(
    tmp_path: Path, bids: dict[int, tuple[int, str]], observe: dict[int, list]
) -> Engine:
    """Run the freeze config; warrior_1 bids for iron at the given ticks.

    ``bids`` maps a tick to (quantity, price). ``observe`` maps a tick to a list
    that receives warrior_1's (frozen, available) gold after that tick's cycle.
    """
    config = _freeze_config()
    assert config.phases.shock_tick == FREEZE_SHOCK_TICK
    assert config.phases.reversion_tick == FREEZE_REVERSION_TICK
    state = build_world(config)
    state.account("warrior_1").gold = Decimal(100)

    def decide(engine: Engine) -> None:
        if engine.tick in bids:
            quantity, price = bids[engine.tick]
            engine.submit_order("warrior_1", OrderSide.BUY, Product.HEALTH_POTION, quantity, price)
        if engine.tick in observe:
            account = engine.state.account("warrior_1")
            observe[engine.tick].append((account.frozen_gold, account.available_gold))

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        for _ in range(config.time.total_cycles):
            engine.run_cycle(decide)
    return engine


@pytest.mark.invariant
@pytest.mark.parametrize("with_bid", [False, True])
def test_a_collection_at_the_shock_tick_runs_before_the_freeze(
    tmp_path: Path, with_bid: bool
) -> None:
    """The wealth tax falls due at the shock tick; the freeze sees what it left owed.

    The collection runs at the cycle start, before the tick's shock. warrior_1
    owes 100 on its 1000 gold: paid at once if the gold is free, owed as arrears
    if it sits in a bid that later expires. Either way half of the 900 left is
    frozen and the other half is available once the bid is gone.
    """
    config = _freeze_config(
        economy={
            "storage_cost_per_unit": {"adventurer": 0, "artisan": 0},
            "taxes": {
                "profit_rate": 0,
                "wealth_rate": "0.1",
                "wealth_every_ticks": FREEZE_SHOCK_TICK,
                "wealth_exemption": 0,
            },
        }
    )
    state = build_world(config)
    state.account("warrior_1").gold = Decimal(1000)
    seen: list[tuple[Decimal, Decimal, Decimal]] = []

    def decide(engine: Engine) -> None:
        if with_bid and engine.tick == FREEZE_SHOCK_TICK - 1:
            engine.submit_order("warrior_1", OrderSide.BUY, Product.HEALTH_POTION, 100, "10")
        if engine.tick == FREEZE_SHOCK_TICK + 5:
            account = engine.state.account("warrior_1")
            seen.append((account.frozen_gold, account.available_gold, account.tax_arrears))

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        for _ in range(FREEZE_SHOCK_TICK + 6):
            engine.run_cycle(decide)

    assert engine.state.account("warrior_1").reserved_gold == ZERO
    assert seen == [(Decimal(450), Decimal(450), ZERO)]


def test_an_agent_with_all_its_gold_in_a_bid_is_unfrozen_at_reversion(tmp_path: Path) -> None:
    """No free gold at the shock, so nothing frozen then -- but it is still affected."""
    observe: dict[int, list] = {FREEZE_SHOCK_TICK: [], FREEZE_REVERSION_TICK: []}
    engine = _run_with_bids(tmp_path, {FREEZE_SHOCK_TICK - 1: (10, "10")}, observe)

    assert observe[FREEZE_SHOCK_TICK] == [(ZERO, ZERO)]
    applied = next(
        e for e in read_events(tmp_path / "run.jsonl") if e.type is EventType.SHOCK_APPLIED
    )
    assert applied.payload["affected_agents"] == ["warrior_1"]
    assert Decimal(applied.payload["frozen_total"]) == ZERO

    # The bid expired during the shock and half of it froze; reversion lifts it.
    assert observe[FREEZE_REVERSION_TICK] == [(ZERO, Decimal(100))]
    account = engine.state.account("warrior_1")
    assert account.frozen_gold == ZERO
    assert account.freeze_rate == ZERO
    assert account.freezable_reserved == ZERO


def test_a_pre_shock_bid_that_expires_during_the_shock_is_frozen_at_the_rate(
    tmp_path: Path,
) -> None:
    observe: dict[int, list] = {FREEZE_SHOCK_TICK + 5: []}
    _run_with_bids(tmp_path, {FREEZE_SHOCK_TICK - 1: (6, "10")}, observe)

    # 40 free at the shock: 20 frozen then. The 60 bid returns at expiry: 30 more.
    assert observe[FREEZE_SHOCK_TICK + 5] == [(Decimal(50), Decimal(50))]
    expired = [
        e
        for e in read_events(tmp_path / "run.jsonl")
        if e.type is EventType.ORDER_EXPIRED and e.agent_id == "warrior_1"
    ]
    assert expired and FREEZE_SHOCK_TICK < expired[0].tick < FREEZE_REVERSION_TICK


def test_a_post_shock_bid_that_expires_is_not_frozen(tmp_path: Path) -> None:
    """Gold bid after the shock was free gold the freeze already applied to."""
    observe: dict[int, list] = {FREEZE_SHOCK_TICK + 1: [], FREEZE_SHOCK_TICK + 8: []}
    _run_with_bids(tmp_path, {FREEZE_SHOCK_TICK + 1: (5, "10")}, observe)

    # At the shock 50 of 100 froze; the post-shock bid took the other 50.
    assert observe[FREEZE_SHOCK_TICK + 1] == [(Decimal(50), ZERO)]
    # It has expired and come back whole: nothing more froze.
    assert observe[FREEZE_SHOCK_TICK + 8] == [(Decimal(50), Decimal(50))]


def test_only_the_pre_freeze_reservation_freezes_when_both_are_released() -> None:
    from rpg_economy.engine.state import AgentAccount

    account = AgentAccount(
        agent_id="b", role=Role.ARTISAN, specialization="smithing", gold=Decimal(100)
    )
    account.reserve_gold(Decimal(40))  # a bid out before the freeze
    account.freeze_gold(Decimal("0.5"), tick=10)  # 30 of the free 60
    account.reserve_gold(Decimal(20))  # a bid placed after it

    account.release_gold(Decimal(20), placed_tick=12)
    assert account.frozen_gold == Decimal(30)
    account.release_gold(Decimal(40), placed_tick=8)
    assert account.frozen_gold == Decimal(50)
    assert account.freezable_reserved == ZERO


def test_a_partial_fill_draws_down_what_can_freeze() -> None:
    """Gold a pre-freeze bid spends or gets back as change can never freeze later."""
    from rpg_economy.engine.state import AgentAccount

    account = AgentAccount(
        agent_id="b", role=Role.ARTISAN, specialization="smithing", gold=Decimal(100)
    )
    account.reserve_gold(Decimal(100))
    account.freeze_gold(Decimal("0.5"), tick=10)
    account.spend_reserved_gold(Decimal(30), placed_tick=5)
    account.refund_gold(Decimal(10), placed_tick=5)
    assert account.freezable_reserved == Decimal(60)

    account.release_gold(Decimal(60), placed_tick=5)
    # Only the release freezes: half of the 60 left. The change came back free.
    assert account.frozen_gold == Decimal(30)
    assert account.available_gold == Decimal(40)
