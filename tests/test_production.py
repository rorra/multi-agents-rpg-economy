"""Crafting and per-cycle capacity (design v1.0 s5.1, s10)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from rpg_economy.domain import ArtisanSpecialization, Product, Resource
from rpg_economy.engine import AgentAccount, ProductionRejected, ProductionSystem, build_world
from tests.conftest import make_config


@pytest.fixture
def system() -> ProductionSystem:
    return ProductionSystem(make_config().economy)


@pytest.fixture
def smith() -> AgentAccount:
    return build_world(make_config()).account("smithing_1")


def stocked(account: AgentAccount, **resources: int) -> AgentAccount:
    for name, quantity in resources.items():
        account.inventory[Resource(name)] = quantity
    return account


def test_crafting_consumes_inputs_and_yields_the_product(
    system: ProductionSystem, smith: AgentAccount
) -> None:
    stocked(smith, iron=10, leather=10)
    result = system.produce(smith, Product.MELEE_WEAPON, 1)

    assert result.quantity == 1
    assert smith.quantity(Product.MELEE_WEAPON) == 1
    assert smith.quantity(Resource.IRON) == 8
    assert smith.quantity(Resource.LEATHER) == 9


def test_cannot_craft_outside_the_specialization(
    system: ProductionSystem, smith: AgentAccount
) -> None:
    stocked(smith, herbs=10)
    with pytest.raises(ProductionRejected, match="requires"):
        system.produce(smith, Product.HEALTH_POTION, 1)


def test_only_artisans_craft(system: ProductionSystem) -> None:
    warrior = build_world(make_config()).account("warrior_1")
    with pytest.raises(ProductionRejected, match="not an artisan"):
        system.produce(warrior, Product.MELEE_WEAPON, 1)


def test_cannot_craft_without_inputs(system: ProductionSystem, smith: AgentAccount) -> None:
    smith.inventory.clear()
    with pytest.raises(ProductionRejected, match="needs"):
        system.produce(smith, Product.MELEE_WEAPON, 1)


def test_reserved_inputs_do_not_count_as_available(
    system: ProductionSystem, smith: AgentAccount
) -> None:
    """Stock promised to a resting sell order cannot also be crafted with."""
    stocked(smith, iron=2, leather=1)
    smith.reserve_quantity(Resource.IRON, 2)
    with pytest.raises(ProductionRejected, match="needs"):
        system.produce(smith, Product.MELEE_WEAPON, 1)


@pytest.mark.parametrize("quantity", [0, -1])
def test_degenerate_quantities_are_rejected(
    system: ProductionSystem, smith: AgentAccount, quantity: int
) -> None:
    with pytest.raises(ProductionRejected, match="positive"):
        system.produce(smith, Product.MELEE_WEAPON, quantity)


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------
def test_capacity_limits_output_per_cycle(system: ProductionSystem, smith: AgentAccount) -> None:
    stocked(smith, iron=100, leather=100)
    capacity = system.capacity_of(smith)
    system.produce(smith, Product.MELEE_WEAPON, capacity)

    assert system.capacity_of(smith) == 0
    with pytest.raises(ProductionRejected, match=r"capacity for .* recipe runs"):
        system.produce(smith, Product.MELEE_WEAPON, 1)


def test_capacity_multiplier_reduces_output(system: ProductionSystem, smith: AgentAccount) -> None:
    """The mechanism of the productivity_drop scenario (design v1.0 s10)."""
    full = system.rate_of(ArtisanSpecialization.SMITHING)
    system.set_capacity_multiplier(ArtisanSpecialization.SMITHING, Decimal("0.3"))
    assert system.rate_of(ArtisanSpecialization.SMITHING) == full * Decimal("0.3")

    smith.capacity_credit = Decimal(0)
    system.refresh(smith)
    assert smith.capacity_credit == full * Decimal("0.3")

    system.clear_capacity_multiplier(ArtisanSpecialization.SMITHING)
    assert system.rate_of(ArtisanSpecialization.SMITHING) == full


def test_a_fractional_rate_accrues_to_one_unit(
    system: ProductionSystem, smith: AgentAccount
) -> None:
    """Below one recipe run per cycle, capacity builds up over several cycles."""
    rate = Decimal("0.4")
    system.set_capacity_multiplier(
        ArtisanSpecialization.SMITHING, rate / system.rate_of(ArtisanSpecialization.SMITHING)
    )
    stocked(smith, iron=100, leather=100)
    smith.capacity_credit = Decimal(0)

    cycles = 0
    while system.capacity_of(smith) == 0:
        system.refresh(smith)
        cycles += 1
    assert cycles == 3  # 0.4, 0.8, 1.2
    system.produce(smith, Product.MELEE_WEAPON, 1)
    assert system.capacity_of(smith) == 0


def test_a_fractional_rate_keeps_its_leftover(
    system: ProductionSystem, smith: AgentAccount
) -> None:
    """A rate of 0.3 crafted at every chance yields 3 runs in 10 cycles, not 2."""
    rate = Decimal("0.3")
    system.set_capacity_multiplier(
        ArtisanSpecialization.SMITHING, rate / system.rate_of(ArtisanSpecialization.SMITHING)
    )
    stocked(smith, iron=100, leather=100)
    smith.capacity_credit = Decimal(0)
    runs = 0
    for _ in range(10):
        system.refresh(smith)
        if system.capacity_of(smith):
            system.produce(smith, Product.MELEE_WEAPON, 1)
            smith.inventory.pop(Product.MELEE_WEAPON)  # sold, so the stock limit stays clear
            runs += 1
    assert runs == 3


def test_an_idle_artisan_never_banks_an_extra_whole_run(smith: AgentAccount) -> None:
    """At a whole-number rate the carry adds no run, however long it waits."""
    config = make_config(economy={"production_capacity_per_cycle": {"smithing": 2}})
    system = ProductionSystem(config.economy)
    smith.capacity_credit = Decimal(0)
    for _ in range(20):
        system.refresh(smith)
    assert system.capacity_of(smith) == 2


def test_a_zero_rate_never_allows_production(system: ProductionSystem, smith: AgentAccount) -> None:
    system.set_capacity_multiplier(ArtisanSpecialization.SMITHING, Decimal(0))
    for _ in range(10):
        system.refresh(smith)
    assert system.capacity_of(smith) == 0


def test_capacity_multiplier_affects_only_its_specialization(
    system: ProductionSystem,
) -> None:
    world = build_world(make_config())
    alchemist = world.account("alchemy_1")
    before = system.capacity_of(alchemist)
    system.set_capacity_multiplier(ArtisanSpecialization.SMITHING, Decimal(0))
    assert system.capacity_of(alchemist) == before


def test_missing_inputs_reports_the_shortfall(
    system: ProductionSystem, smith: AgentAccount
) -> None:
    smith.inventory.clear()
    stocked(smith, iron=1)
    assert system.missing_inputs(smith, Product.MELEE_WEAPON) == {
        Resource.IRON: 1,
        Resource.LEATHER: 1,
    }


def test_no_missing_inputs_when_stocked(system: ProductionSystem, smith: AgentAccount) -> None:
    stocked(smith, iron=5, leather=5)
    assert system.missing_inputs(smith, Product.MELEE_WEAPON) == {}


def test_products_are_partitioned_by_specialization(system: ProductionSystem) -> None:
    smithing = set(system.products_for(ArtisanSpecialization.SMITHING))
    alchemy = set(system.products_for(ArtisanSpecialization.ALCHEMY))

    assert smithing == {Product.MELEE_WEAPON, Product.ARMOR}
    assert not smithing & alchemy


def test_every_product_has_exactly_one_recipe() -> None:
    recipes = make_config().economy.recipes
    assert set(recipes) == set(Product)


def test_the_base_recipes_are_the_calibrated_ones() -> None:
    """Pins the calibrated recipe table, so a change to it is a deliberate one."""
    recipes = make_config().economy.recipes
    table = {
        product.value: ({good.value: int(n) for good, n in r.inputs.items()}, r.output_quantity)
        for product, r in recipes.items()
    }
    assert table == {
        "melee_weapon": ({"iron": 2, "leather": 1}, 1),
        "armor": ({"iron": 2, "leather": 2}, 1),
        "bow_gear": ({"wood": 2, "leather": 2}, 1),
        "staff_focus": ({"wood": 2, "arcane_crystal": 2}, 1),
        "health_potion": ({"herbs": 1}, 1),
        "mana_potion": ({"herbs": 1, "arcane_crystal": 1}, 2),
    }


def test_an_artisan_holding_the_stock_limit_cannot_craft_more(
    system: ProductionSystem, smith: AgentAccount
) -> None:
    """Woodworkers made 23 bows for 18 sales; units on its own sell
    orders count toward the limit."""
    limit = make_config().economy.craft_stock_limit
    assert limit == 2
    stocked(smith, iron=100, leather=100)
    smith.receive(Product.MELEE_WEAPON, limit - 1)
    system.produce(smith, Product.MELEE_WEAPON, 1)  # below the limit: allowed
    system.refresh(smith)
    smith.capacity_credit = Decimal(5)
    with pytest.raises(ProductionRejected, match="until fewer than 2 remain"):
        system.produce(smith, Product.MELEE_WEAPON, 1)
    # Another product is not held back.
    system.produce(smith, Product.ARMOR, 1)
