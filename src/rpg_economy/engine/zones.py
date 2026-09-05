"""Expedition zones and resource appearance (design v1.0 s5.3, s10).

Raw materials enter the economy in exactly one way: an adventurer brings them
back from a zone. Each zone accrues resources at a configured rate, capped at a
finite pool, and expeditions draw that pool down.

The cap is what makes the primary scenario work. The ``resource_scarcity`` shock
"reduces the appearance of iron" (design s10); against an unbounded stock a lower
accrual rate would be absorbed by whatever had already piled up, and the shock
would take effect only long after its reversion tick. A pool that saturates at a
few ticks' worth of supply means a reduced rate is felt while it is in force.

Accrual uses an exact ``Decimal`` accumulator rather than rounding each tick, so
a rate of 0.2/tick delivers exactly one unit every five ticks, deterministically,
instead of being silently truncated to nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal

from rpg_economy.config.models import ExpeditionConfig
from rpg_economy.domain import Resource, RiskLevel

ZERO = Decimal(0)
ONE = Decimal(1)


@dataclass
class Zone:
    """One zone's difficulty, appearance rates and current stock."""

    zone_id: str
    risk: RiskLevel
    spawn_per_tick: dict[Resource, Decimal]
    capacity: dict[Resource, int]
    available: dict[Resource, int] = field(default_factory=dict)
    progress: dict[Resource, Decimal] = field(default_factory=dict)
    #: Gold appearing per tick, and the most the zone holds; None: unlimited.
    gold_rate: Decimal | None = None
    gold_cap: Decimal | None = None
    gold: Decimal = ZERO

    def take_gold(self, amount: Decimal) -> Decimal:
        """Remove up to ``amount`` gold; return how much was actually there."""
        if self.gold_rate is None:
            return amount
        taken = min(amount, self.gold)
        self.gold -= taken
        return taken

    def stock(self, resource: Resource) -> int:
        return self.available.get(resource, 0)

    def take(self, resource: Resource, quantity: int) -> int:
        """Remove up to ``quantity`` units; return how many were actually there."""
        taken = min(quantity, self.stock(resource))
        if taken > 0:
            self.available[resource] = self.stock(resource) - taken
        return taken


class ZoneSystem:
    """All zones, and the per-tick appearance of resources within them."""

    def __init__(self, config: ExpeditionConfig) -> None:
        self._zones: dict[str, Zone] = {}
        for zone_config in config.zones:
            # Rounded up: a slow resource (0.1 per tick over 6 ticks) must still
            # be able to hold one unit, or it could never appear at all.
            capacity = {
                resource: int((rate * config.pool_capacity_ticks).to_integral_value(ROUND_CEILING))
                for resource, rate in zone_config.spawn_per_tick.items()
            }
            gold_rate = zone_config.gold_per_tick
            gold_ticks = config.gold_capacity_ticks or config.pool_capacity_ticks
            gold_cap = None if gold_rate is None else gold_rate * gold_ticks
            self._zones[zone_config.id] = Zone(
                zone_id=zone_config.id,
                risk=zone_config.risk,
                spawn_per_tick=dict(zone_config.spawn_per_tick),
                capacity=capacity,
                gold_rate=gold_rate,
                gold_cap=gold_cap,
                gold=gold_cap if gold_cap is not None else ZERO,
                # Zones start full, so the base phase is not a supply ramp-up.
                available=dict(capacity),
                progress=dict.fromkeys(zone_config.spawn_per_tick, ZERO),
            )
        #: Per-resource multiplier on appearance, driven by shocks. 1 = normal.
        self._spawn_multiplier: dict[Resource, Decimal] = {}

    # -- access ------------------------------------------------------------
    @property
    def zone_ids(self) -> list[str]:
        """Zone ids in configuration order, which is fixed and reproducible."""
        return list(self._zones)

    def zone(self, zone_id: str) -> Zone:
        try:
            return self._zones[zone_id]
        except KeyError:
            raise KeyError(f"unknown zone {zone_id!r}") from None

    def zones(self) -> list[Zone]:
        return list(self._zones.values())

    def total_available(self, resource: Resource) -> int:
        return sum(zone.stock(resource) for zone in self._zones.values())

    # -- shock hooks -------------------------------------------------------
    def spawn_multiplier(self, resource: Resource) -> Decimal:
        return self._spawn_multiplier.get(resource, ONE)

    def set_spawn_multiplier(self, resource: Resource, multiplier: Decimal) -> None:
        self._spawn_multiplier[resource] = multiplier

    def clear_spawn_multiplier(self, resource: Resource) -> None:
        self._spawn_multiplier.pop(resource, None)

    def scale_stock(self, resource: Resource, multiplier: Decimal) -> None:
        """Cut what every pool holds of ``resource`` to ``multiplier`` of it, rounded down."""
        for zone in self._zones.values():
            if resource in zone.available:
                zone.available[resource] = int(zone.stock(resource) * multiplier)

    # -- per-tick appearance -----------------------------------------------
    def accrue(self) -> None:
        """Advance every zone's stock by one tick of appearance."""
        for zone in self._zones.values():
            # Gold is not a resource: no scarcity shock touches it.
            if zone.gold_rate is not None and zone.gold_cap is not None:
                zone.gold = min(zone.gold + zone.gold_rate, zone.gold_cap)
            for resource in sorted(zone.spawn_per_tick, key=lambda r: r.value):
                rate = zone.spawn_per_tick[resource] * self.spawn_multiplier(resource)
                if rate <= ZERO:
                    continue
                accumulated = zone.progress.get(resource, ZERO) + rate
                whole = int(accumulated)
                zone.progress[resource] = accumulated - whole
                if whole:
                    capped = min(zone.stock(resource) + whole, zone.capacity.get(resource, 0))
                    zone.available[resource] = capped
