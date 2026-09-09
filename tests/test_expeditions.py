"""Expeditions: risk levels, subclass and equipment effects, seeded outcomes."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from rpg_economy.domain import AdventurerSubclass, ExpeditionOutcome, Product, Resource, RiskLevel
from rpg_economy.engine import (
    AgentAccount,
    ExpeditionRejected,
    ExpeditionSystem,
    RunRandom,
    ZoneSystem,
    build_world,
)
from rpg_economy.engine.expeditions import PROBABILITY_SCALE
from tests.conftest import make_config

IRON = Resource.IRON


@pytest.fixture
def system() -> ExpeditionSystem:
    config = make_config()
    return ExpeditionSystem(
        config.expeditions,
        ZoneSystem(config.expeditions),
        frozenset(config.economy.durability),
    )


@pytest.fixture
def warrior() -> AgentAccount:
    return build_world(make_config()).account("warrior_1")


class AlwaysSucceeds(random.Random):
    """A generator whose probability draw always lands below any threshold."""

    def randrange(self, *args: int) -> int:  # type: ignore[override]
        return 0

    def randint(self, a: int, b: int) -> int:  # type: ignore[override]
        return a


class AlwaysFails(random.Random):
    def randrange(self, *args: int) -> int:  # type: ignore[override]
        return args[0] - 1 if args else 0

    def randint(self, a: int, b: int) -> int:  # type: ignore[override]
        return b


# ---------------------------------------------------------------------------
# Starting
# ---------------------------------------------------------------------------
def test_starting_sets_a_resolution_tick_from_zone_risk(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    state, _ = system.start(warrior, "arcane_deeps", tick=5)
    assert state.risk is RiskLevel.HIGH
    assert state.resolves_at_tick == 5 + make_config().expeditions.duration_ticks[RiskLevel.HIGH]


def test_only_adventurers_go_on_expeditions(system: ExpeditionSystem) -> None:
    artisan = build_world(make_config()).account("woodworking_1")
    with pytest.raises(ExpeditionRejected, match="not an adventurer"):
        system.start(artisan, "green_hollow", tick=0)


def test_cannot_start_while_already_away(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    system.start(warrior, "green_hollow", tick=0)
    with pytest.raises(ExpeditionRejected, match="already away"):
        system.start(warrior, "green_hollow", tick=1)


def test_cannot_start_while_recovering(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    warrior.recovering_until_tick = 10
    with pytest.raises(ExpeditionRejected, match=r"is resting .* before tick 10"):
        system.start(warrior, "green_hollow", tick=5)
    state, _ = system.start(warrior, "green_hollow", tick=10)
    assert state.started_tick == 10


# ---------------------------------------------------------------------------
# Success probability: zone, subclass, equipment
# ---------------------------------------------------------------------------
def test_higher_risk_zones_are_less_likely_to_succeed(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    probabilities = [system.success_probability(warrior, risk) for risk in RiskLevel]
    assert probabilities == sorted(probabilities, reverse=True)


def test_subclass_advantage_raises_the_probability(system: ExpeditionSystem) -> None:
    """Warriors are built for dangerous zones (design v1.0 s4.1)."""
    world = build_world(make_config())
    warrior = world.account("warrior_1")
    mage = world.account("mage_1")
    assert system.success_probability(warrior, RiskLevel.HIGH) > system.success_probability(
        mage, RiskLevel.HIGH
    )


def test_equipment_raises_the_probability(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    gear = make_config().expeditions.gear[AdventurerSubclass.WARRIOR]
    warrior.equipment.clear()
    bare = system.success_probability(warrior, RiskLevel.MEDIUM)
    warrior.equipment[Product.MELEE_WEAPON] = 20
    assert system.success_probability(warrior, RiskLevel.MEDIUM) == (
        bare + gear[Product.MELEE_WEAPON]
    )


def test_broken_equipment_confers_no_bonus(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    warrior.equipment.clear()
    bare = system.success_probability(warrior, RiskLevel.MEDIUM)
    warrior.equipment[Product.MELEE_WEAPON] = 0
    assert system.success_probability(warrior, RiskLevel.MEDIUM) == bare


def test_probability_is_clamped_within_bounds(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    """Nothing is ever certain in either direction."""
    config = make_config().expeditions
    for risk in RiskLevel:
        warrior.equipment[Product.MELEE_WEAPON] = 50
        probability = system.success_probability(warrior, risk)
        assert config.min_success_probability <= probability <= config.max_success_probability


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def test_success_yields_resources_and_only_the_rest(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    system.start(warrior, "iron_marches", tick=0)
    result = system.resolve(warrior, AlwaysSucceeds(1), tick=3)

    assert result.outcome is ExpeditionOutcome.SUCCESS
    assert result.total_units > 0
    assert result.recovering_until_tick == 3 + make_config().expeditions.rest_ticks
    assert warrior.expedition is None


def test_failure_yields_less_and_costs_recovery_time(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    """Failure is partial loss and recovery, never death (design s5.3, s14)."""
    system.start(warrior, "iron_marches", tick=0)
    result = system.resolve(warrior, AlwaysFails(1), tick=3)

    config = make_config().expeditions
    assert result.outcome is ExpeditionOutcome.FAILURE
    assert result.recovering_until_tick == (
        3 + config.rest_ticks + config.recovery_ticks_on_failure[RiskLevel.MEDIUM]
    )
    assert config.recovery_ticks_on_failure[RiskLevel.MEDIUM] > 0, "a failure costs more rest"
    assert warrior.is_away(tick=3)
    assert result.total_units < config.expected_yield[RiskLevel.MEDIUM]


def test_higher_risk_carries_a_higher_expected_reward(system: ExpeditionSystem) -> None:
    config = make_config().expeditions
    yields = [config.expected_yield[risk] for risk in RiskLevel]
    spreads = [config.yield_spread[risk] for risk in RiskLevel]
    assert yields == sorted(yields)
    assert spreads == sorted(spreads), "higher risk must also carry higher variance"


def test_yield_lands_in_the_adventurers_inventory(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    system.start(warrior, "iron_marches", tick=0)
    result = system.resolve(warrior, AlwaysSucceeds(1), tick=3)
    for resource, quantity in result.yielded.items():
        assert warrior.quantity(resource) >= quantity


def test_yield_depletes_the_zone_pool(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    config = make_config()
    zones = ZoneSystem(config.expeditions)
    system = ExpeditionSystem(config.expeditions, zones, frozenset(config.economy.durability))
    before = sum(zones.zone("iron_marches").available.values())

    system.start(warrior, "iron_marches", tick=0)
    result = system.resolve(warrior, AlwaysSucceeds(1), tick=3)

    after = sum(zones.zone("iron_marches").available.values())
    assert before - after == result.total_units


def test_an_exhausted_zone_yields_nothing(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    config = make_config()
    zones = ZoneSystem(config.expeditions)
    system = ExpeditionSystem(config.expeditions, zones, frozenset(config.economy.durability))
    zone = zones.zone("green_hollow")
    for resource in list(zone.available):
        zone.take(resource, zone.stock(resource))

    system.start(warrior, "green_hollow", tick=0)
    result = system.resolve(warrior, AlwaysSucceeds(1), tick=2)
    assert result.total_units == 0


def test_subclass_affinity_shifts_the_resource_mix() -> None:
    """A mage brings back more crystal than a warrior from the same zone.

    Each draw uses a fresh zone. Reusing one would let the pool cap bind before
    affinity could show, and the test would measure depletion rather than the
    subclass advantage design s4.1 specifies.
    """
    config = make_config()

    def haul(agent_id: str) -> int:
        total = 0
        for seed in range(200):
            zones = ZoneSystem(config.expeditions)
            system = ExpeditionSystem(
                config.expeditions, zones, frozenset(config.economy.durability)
            )
            account = build_world(config).account(agent_id)
            system.start(account, "arcane_deeps", tick=0)
            result = system.resolve(account, random.Random(seed), tick=4)
            total += result.yielded.get(Resource.ARCANE_CRYSTAL, 0)
        return total

    mage, warrior = haul("mage_1"), haul("warrior_1")
    assert mage > warrior, f"mage gathered {mage} crystal, warrior {warrior}"


def test_expedition_wears_equipment(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    warrior.equipment[Product.MELEE_WEAPON] = 10
    system.start(warrior, "green_hollow", tick=0)
    system.resolve(warrior, AlwaysSucceeds(1), tick=2)

    loss = make_config().expeditions.durability_loss_per_expedition
    assert warrior.durability_of(Product.MELEE_WEAPON) == 10 - loss


def test_equipment_breaks_when_worn_out(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    warrior.equipment[Product.MELEE_WEAPON] = 1
    system.start(warrior, "green_hollow", tick=0)
    result = system.resolve(warrior, AlwaysSucceeds(1), tick=2)

    assert result.equipment_broken == [Product.MELEE_WEAPON]
    assert Product.MELEE_WEAPON not in warrior.equipment


# ---------------------------------------------------------------------------
# Determinism of outcomes
# ---------------------------------------------------------------------------
def test_outcomes_are_reproducible_from_the_run_seed() -> None:
    def haul() -> list[tuple[str, int]]:
        config = make_config()
        zones = ZoneSystem(config.expeditions)
        system = ExpeditionSystem(config.expeditions, zones, frozenset(config.economy.durability))
        world = build_world(config)
        run = RunRandom(config.run.seed)
        results = []
        for _ in range(15):
            account = world.account("ranger_1")
            account.expedition = None
            account.recovering_until_tick = 0
            account.equipment[Product.BOW_GEAR] = 99  # never breaks within the loop
            system.start(account, "green_hollow", tick=0)
            result = system.resolve(account, run.agent("ranger_1"), tick=2)
            results.append((result.outcome.value, result.total_units))
        return results

    assert haul() == haul()


def test_probability_draws_avoid_float_entirely(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    """A Decimal probability compared against a float draw would be ambiguous."""
    probability = system.success_probability(warrior, RiskLevel.LOW)
    assert isinstance(probability, Decimal)
    assert isinstance(int(probability * PROBABILITY_SCALE), int)


def test_each_adventurer_draws_from_its_own_stream() -> None:
    """Activation order must not change what an agent draws (seed-trio pairing)."""
    run = RunRandom(11)
    baseline = [run.agent("warrior_1").random() for _ in range(3)]

    other = RunRandom(11)
    for _ in range(25):
        other.agent("mage_1").random()

    assert [other.agent("warrior_1").random() for _ in range(3)] == baseline


def test_every_subclass_has_its_gear() -> None:
    """Armor fits everyone; each subclass has its weapon; potions for all (design s4.1)."""
    gear = make_config().expeditions.gear
    assert set(gear) == set(AdventurerSubclass)
    assert set(gear[AdventurerSubclass.WARRIOR]) == {
        Product.MELEE_WEAPON,
        Product.ARMOR,
        Product.HEALTH_POTION,
    }
    assert set(gear[AdventurerSubclass.RANGER]) == {
        Product.BOW_GEAR,
        Product.ARMOR,
        Product.HEALTH_POTION,
    }
    mage = gear[AdventurerSubclass.MAGE]
    assert set(mage) == {
        Product.STAFF_FOCUS,
        Product.ARMOR,
        Product.HEALTH_POTION,
        Product.MANA_POTION,
    }
    assert max(mage, key=mage.get) is Product.MANA_POTION


def test_every_equipped_item_adds_its_own_bonus(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    gear = make_config().expeditions.gear[AdventurerSubclass.WARRIOR]
    warrior.equipment.clear()
    bare = system.success_probability(warrior, RiskLevel.HIGH)
    warrior.equipment[Product.MELEE_WEAPON] = 20
    warrior.equipment[Product.ARMOR] = 20
    assert system.success_probability(warrior, RiskLevel.HIGH) == (
        bare + gear[Product.MELEE_WEAPON] + gear[Product.ARMOR]
    )


def test_an_item_outside_the_gear_adds_nothing_but_still_wears(
    system: ExpeditionSystem,
) -> None:
    """A mage wielding a warrior's sword gets no bonus from it."""
    mage = build_world(make_config()).account("mage_1")
    bare = system.success_probability(mage, RiskLevel.MEDIUM)
    mage.equipment[Product.MELEE_WEAPON] = 1
    assert system.success_probability(mage, RiskLevel.MEDIUM) == bare

    system.start(mage, "green_hollow", tick=0)
    result = system.resolve(mage, AlwaysSucceeds(1), tick=2)
    assert result.equipment_broken == [Product.MELEE_WEAPON]


# ---------------------------------------------------------------------------
# Potions: carried, used up, and fixed at departure (design s5.3)
# ---------------------------------------------------------------------------
def _no_surge(_product: Product) -> Decimal:
    return Decimal(1)


def test_a_potion_adds_its_bonus_and_is_used_up(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    gear = make_config().expeditions.gear[AdventurerSubclass.WARRIOR]
    warrior.inventory[Product.HEALTH_POTION] = 3
    needed = system.supplies_for(warrior, _no_surge)
    assert needed == {Product.HEALTH_POTION: 1}

    state, consumed = system.start(warrior, "green_hollow", tick=0, supplies=needed)
    assert consumed == {Product.HEALTH_POTION: 1}
    assert warrior.quantity(Product.HEALTH_POTION) == 2
    assert state.gear_bonus == {
        Product.HEALTH_POTION: gear[Product.HEALTH_POTION],
        Product.MELEE_WEAPON: gear[Product.MELEE_WEAPON],  # worn from the start
    }


def test_a_mage_takes_mana_as_well() -> None:
    config = make_config()
    system = ExpeditionSystem(
        config.expeditions,
        ZoneSystem(config.expeditions),
        frozenset(config.economy.durability),
    )
    mage = build_world(config).account("mage_1")
    assert system.supplies_for(mage, _no_surge) == {
        Product.HEALTH_POTION: 1,
        Product.MANA_POTION: 1,
    }


def test_setting_out_without_potions_just_forgoes_their_bonus(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    warrior.inventory.pop(Product.HEALTH_POTION, None)
    needed = system.supplies_for(warrior, _no_surge)
    state, consumed = system.start(warrior, "green_hollow", tick=0, supplies=needed)

    assert consumed == {}
    assert Product.HEALTH_POTION not in state.gear_bonus
    assert system.success_probability(
        warrior, RiskLevel.LOW, gear_bonus=state.gear_bonus
    ) == system.success_probability(warrior, RiskLevel.LOW)


def test_under_a_surge_a_short_supply_gives_a_share_of_the_bonus(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    gear = make_config().expeditions.gear[AdventurerSubclass.WARRIOR]
    warrior.inventory[Product.HEALTH_POTION] = 1
    needed = system.supplies_for(warrior, lambda _p: Decimal("3"))
    assert needed == {Product.HEALTH_POTION: 3}

    state, consumed = system.start(warrior, "green_hollow", tick=0, supplies=needed)
    assert consumed == {Product.HEALTH_POTION: 1}
    assert state.gear_bonus[Product.HEALTH_POTION] == gear[Product.HEALTH_POTION] * (Decimal(1) / 3)


def test_a_surge_rounds_the_supplies_up(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    needed = system.supplies_for(warrior, lambda _p: Decimal("1.5"))
    assert needed == {Product.HEALTH_POTION: 2}


def test_the_odds_are_fixed_at_departure(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    """Gear that breaks on the road still counted: it was carried when leaving."""
    warrior.equipment[Product.MELEE_WEAPON] = 20
    state, _ = system.start(warrior, "iron_marches", tick=0)
    warrior.equipment.clear()
    assert Product.MELEE_WEAPON in state.gear_bonus
    fixed = system.success_probability(warrior, RiskLevel.MEDIUM, gear_bonus=state.gear_bonus)
    assert fixed > system.success_probability(warrior, RiskLevel.MEDIUM)


@pytest.mark.invariant
def test_an_expeditions_luck_depends_only_on_who_where_and_when() -> None:
    """Paired trios: other draws from the agent's stream must not shift the outcome."""
    from rpg_economy.engine import RunRandom

    first = RunRandom(7)
    second = RunRandom(7)
    for _ in range(50):
        second.agent("warrior_1").random()  # decisions taken in one condition only

    a = first.expedition("warrior_1", "iron_marches", 12).random()
    b = second.expedition("warrior_1", "iron_marches", 12).random()
    assert a == b
    assert a != first.expedition("warrior_1", "iron_marches", 13).random()


# ---------------------------------------------------------------------------
# Gold loot and wear per trip
# ---------------------------------------------------------------------------
def test_a_successful_trip_brings_back_gold_by_zone_risk(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    config = make_config()
    before = warrior.gold
    state, _ = system.start(warrior, "arcane_deeps", tick=0)
    result = system.resolve(warrior, AlwaysSucceeds(1), tick=state.resolves_at_tick)

    assert result.outcome is ExpeditionOutcome.SUCCESS
    # Its weapon, carried, adds its share (gear_gold_bonus).
    share = config.expeditions.gear_gold_bonus[Product.MELEE_WEAPON]
    assert result.gold == config.expeditions.gold_loot[RiskLevel.HIGH] * (1 + share)
    assert result.gold > 0
    assert warrior.gold == before + result.gold


def test_a_failed_trip_brings_back_no_gold(system: ExpeditionSystem, warrior: AgentAccount) -> None:
    before = warrior.gold
    state, _ = system.start(warrior, "arcane_deeps", tick=0)
    result = system.resolve(warrior, AlwaysFails(1), tick=state.resolves_at_tick)

    assert result.outcome is ExpeditionOutcome.FAILURE
    assert result.gold == 0
    assert warrior.gold == before


def test_zero_loot_means_no_gold() -> None:
    config = make_config(expeditions={"gold_loot": {"low": 0, "medium": 0, "high": 0}})
    system = ExpeditionSystem(
        config.expeditions, ZoneSystem(config.expeditions), frozenset(config.economy.durability)
    )
    warrior = build_world(config).account("warrior_1")
    before = warrior.gold
    system.start(warrior, "green_hollow", tick=0)
    result = system.resolve(warrior, AlwaysSucceeds(1), tick=5)

    assert result.gold == 0
    assert warrior.gold == before


def test_every_trip_costs_the_configured_wear(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    loss = make_config().expeditions.durability_loss_per_expedition
    warrior.equipment[Product.MELEE_WEAPON] = 10
    for trip in range(3):
        warrior.recovering_until_tick = 0
        state, _ = system.start(warrior, "green_hollow", tick=trip * 10)
        system.resolve(warrior, AlwaysSucceeds(1), tick=state.resolves_at_tick)
        assert warrior.durability_of(Product.MELEE_WEAPON) == 10 - loss * (trip + 1)


def test_a_returned_adventurer_rests_before_setting_out_again(
    system: ExpeditionSystem, warrior: AgentAccount
) -> None:
    """It is back in town and can trade, but cannot leave until the rest is over."""
    rest = make_config().expeditions.rest_ticks
    system.start(warrior, "iron_marches", tick=0)
    system.resolve(warrior, AlwaysSucceeds(1), tick=3)

    assert warrior.is_resting(tick=3)
    assert warrior.expedition is None
    with pytest.raises(ExpeditionRejected):
        system.start(warrior, "iron_marches", tick=3 + rest - 1)
    system.start(warrior, "iron_marches", tick=3 + rest)
    assert warrior.expedition is not None


def test_a_mage_brings_back_more_gold_per_trip() -> None:
    config = make_config()
    system = ExpeditionSystem(config.expeditions, ZoneSystem(config.expeditions), frozenset())
    world = build_world(config)
    mage, warrior = world.account("mage_1"), world.account("warrior_1")
    multiplier = config.expeditions.subclasses[AdventurerSubclass.MAGE].loot_multiplier
    assert multiplier > 1
    gold = {}
    for account in (mage, warrior):
        before = account.gold
        system.start(account, "iron_marches", tick=0)
        system.resolve(account, AlwaysSucceeds(1), tick=3)
        gold[account.agent_id] = account.gold - before
    assert gold["mage_1"] == gold["warrior_1"] * multiplier
