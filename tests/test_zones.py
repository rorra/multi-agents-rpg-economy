"""Zones and resource appearance (design v1.0 s5.3)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from rpg_economy.domain import Resource
from rpg_economy.engine import ZoneSystem
from tests.conftest import make_config

IRON = Resource.IRON


@pytest.fixture
def zones() -> ZoneSystem:
    return ZoneSystem(make_config().expeditions)


def test_zones_come_from_configuration(zones: ZoneSystem) -> None:
    assert zones.zone_ids == ["green_hollow", "iron_marches", "arcane_deeps"]


def test_zones_start_full(zones: ZoneSystem) -> None:
    """The base phase should observe a working economy, not a supply ramp-up."""
    for zone in zones.zones():
        for resource, capacity in zone.capacity.items():
            assert zone.stock(resource) == capacity


def test_accrual_is_capped_at_capacity(zones: ZoneSystem) -> None:
    before = zones.total_available(IRON)
    for _ in range(500):
        zones.accrue()
    assert zones.total_available(IRON) == before


def test_fractional_rates_accrue_exactly() -> None:
    """A rate of 1/n per tick must deliver 1 unit every n ticks, not round to nothing."""
    config = make_config()
    payload = config.expeditions.model_dump(mode="python")
    for zone_config in payload["zones"]:
        if zone_config["id"] == "green_hollow":
            zone_config["spawn_per_tick"][IRON] = Decimal("0.125")
    zones = ZoneSystem(type(config.expeditions).model_validate(payload))
    zone = zones.zone("green_hollow")
    rate = zone.spawn_per_tick[IRON]
    assert Decimal(0) < rate < Decimal(1)
    every = int(Decimal(1) / rate)
    assert Decimal(every) * rate == Decimal(1)  # the fixture rate divides one exactly
    zone.take(IRON, zone.stock(IRON))

    for _ in range(every - 1):
        zones.accrue()
    assert zone.stock(IRON) == 0

    zones.accrue()
    assert zone.stock(IRON) == 1


def test_taking_is_bounded_by_stock(zones: ZoneSystem) -> None:
    zone = zones.zone("iron_marches")
    available = zone.stock(IRON)
    assert zone.take(IRON, available + 100) == available
    assert zone.stock(IRON) == 0


def test_spawn_multiplier_reduces_appearance(zones: ZoneSystem) -> None:
    """This is the mechanism of the primary scenario (design v1.0 s10)."""
    for zone in zones.zones():
        zone.take(IRON, zone.stock(IRON))

    zones.set_spawn_multiplier(IRON, Decimal("0.25"))
    for _ in range(20):
        zones.accrue()
    scarce = zones.total_available(IRON)

    zones.clear_spawn_multiplier(IRON)
    for zone in zones.zones():
        zone.take(IRON, zone.stock(IRON))
    for _ in range(20):
        zones.accrue()

    assert scarce < zones.total_available(IRON)


def test_a_zero_multiplier_stops_appearance_entirely(zones: ZoneSystem) -> None:
    for zone in zones.zones():
        zone.take(IRON, zone.stock(IRON))
    zones.set_spawn_multiplier(IRON, Decimal(0))
    for _ in range(50):
        zones.accrue()
    assert zones.total_available(IRON) == 0


def test_a_multiplier_affects_only_its_resource(zones: ZoneSystem) -> None:
    for zone in zones.zones():
        for resource in (IRON, Resource.WOOD):
            zone.take(resource, zone.stock(resource))
    zones.set_spawn_multiplier(IRON, Decimal(0))
    for _ in range(20):
        zones.accrue()

    assert zones.total_available(IRON) == 0
    assert zones.total_available(Resource.WOOD) > 0


def test_unknown_zone_is_an_error(zones: ZoneSystem) -> None:
    with pytest.raises(KeyError, match="unknown zone"):
        zones.zone("nowhere")


def test_a_slow_resource_can_still_hold_a_unit() -> None:
    """Capacity rounds up: a rate that fills less than one unit per window still appears."""
    from rpg_economy.engine import ZoneSystem
    from tests.conftest import make_config

    config = make_config()
    zones = ZoneSystem(config.expeditions)
    for zone in zones.zones():
        for resource, rate in zone.spawn_per_tick.items():
            if rate > 0:
                assert zone.capacity[resource] >= 1
