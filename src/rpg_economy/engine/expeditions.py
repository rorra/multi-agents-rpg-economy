"""Expeditions: the only way raw materials enter the economy (design v1.0 s5.3).

An expedition takes ticks to complete and resolves against the adventurer's own
seeded stream. Success and reward depend on zone difficulty, subclass and
equipment, exactly as s5.3 requires; higher risk carries both a higher expected
reward and a wider spread. Failure costs recovery time and yields a fraction of
the reward -- never permanent death, which s14 excludes.

All randomness here is integer-based. Probabilities and weights are
``Decimal`` in configuration, and drawing with ``random.random()`` would compare a
float against a Decimal and pull float rounding into the outcome of every
expedition. Instead a probability p is realised as ``randrange(SCALE) <
int(p * SCALE)``, which is exact, reproducible, and unambiguous at the boundary.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal

from rpg_economy.config.models import ExpeditionConfig
from rpg_economy.domain import (
    AdventurerSubclass,
    ExpeditionOutcome,
    Product,
    Resource,
    RiskLevel,
    Role,
)
from rpg_economy.engine.state import AgentAccount, ExpeditionState
from rpg_economy.engine.zones import Zone, ZoneSystem

ZERO = Decimal(0)
ONE = Decimal(1)

#: Denominator for integer probability draws. One part per million is finer than
#: any calibrated probability and keeps every comparison exact.
PROBABILITY_SCALE = 1_000_000

#: Denominator for integer weight draws in the resource distribution.
WEIGHT_SCALE = 1_000


class ExpeditionRejected(Exception):
    """Raised when an expedition cannot be started."""


@dataclass(frozen=True)
class ExpeditionResult:
    """What an expedition returned with."""

    agent_id: str
    zone_id: str
    risk: RiskLevel
    outcome: ExpeditionOutcome
    success_probability: Decimal
    yielded: dict[Resource, int]
    recovering_until_tick: int
    equipment_broken: list[Product]
    gold: Decimal = Decimal(0)
    #: The gold the success was due before the zone's gold pool was shared.
    gold_due: Decimal = Decimal(0)

    @property
    def total_units(self) -> int:
        return sum(self.yielded.values())


class ExpeditionSystem:
    """Starts and resolves expeditions."""

    def __init__(
        self,
        config: ExpeditionConfig,
        zones: ZoneSystem,
        durable: frozenset[Product] | set[Product],
    ) -> None:
        """``durable`` names the products that are equipped and wear out."""
        self._config = config
        self._zones = zones
        self._durable = frozenset(durable)
        #: (tick, zone_id) of every expedition started: what crowding counts.
        self._starts: list[tuple[int, str]] = []

    # -- crowding ----------------------------------------------------------
    def traffic(self, tick: int) -> dict[str, int]:
        """Expeditions set out for each zone over the crowding window up to ``tick``.

        Public, like adventurers seen leaving for a zone: who went, not what they
        found. Every zone is listed, at 0 when none went.
        """
        counts = dict.fromkeys(self._zones.zone_ids, 0)
        crowding = self._config.crowding
        if crowding is None:
            return counts
        since = tick - crowding.window_ticks
        for started, zone_id in self._starts:
            if started > since:
                counts[zone_id] += 1
        return counts

    def likely_haul(self, zone_id: str, tick: int) -> Decimal:
        """Units a success there is likely to find, from public facts only.

        The least of the full haul, a full pool (what the zone grows over the
        pool's ticks), and what it grows over the crowding window shared among
        the trips that went there: a zone everyone uses is picked clean, and its
        trips come back with what grew since the last.
        """
        zone = self._zones.zone(zone_id)
        rate = sum(zone.spawn_per_tick.values(), ZERO)
        # Never more than a full pool holds: with the spawn set to what the
        # economy uses, a hard zone's full pool is under half its full haul.
        full = min(
            Decimal(self._config.expected_yield[zone.risk]),
            rate * self._config.pool_capacity_ticks,
        )
        settings = self._config.crowding
        trips = self.traffic(tick)[zone_id] if settings is not None else 0
        if trips:
            full = min(full, rate * settings.window_ticks / trips)
        return full.quantize(Decimal("0.1"))

    def full_gold(
        self, account: AgentAccount, zone: Zone, bonus: dict[Product, Decimal]
    ) -> Decimal:
        """A success's gold in ``zone`` carrying ``bonus``'s items, before any sharing."""
        gold = self._config.gold_loot.get(zone.risk, ZERO)
        subclass = self._subclass_of(account)
        settings = self._config.subclasses.get(subclass) if subclass is not None else None
        if settings is not None:
            gold *= settings.loot_multiplier
        return gold * (
            ONE
            + sum(
                (self._config.gear_gold_bonus.get(p, ZERO) for p, b in bonus.items() if b > ZERO),
                ZERO,
            )
        )

    def likely_gold(self, zone_id: str, tick: int, full: Decimal) -> Decimal:
        """The gold a success there is likely to bring, from public facts only.

        The least of its ``full`` gold, a full gold pool, and the gold the zone
        grows over the crowding window shared among the trips that went there.
        """
        zone = self._zones.zone(zone_id)
        if zone.gold_rate is None or zone.gold_cap is None:
            return full
        likely = min(full, zone.gold_cap)
        settings = self._config.crowding
        trips = self.traffic(tick)[zone_id] if settings is not None else 0
        if trips:
            likely = min(likely, zone.gold_rate * settings.window_ticks / trips)
        return likely.quantize(Decimal(1))

    def crowding(self, zone_id: str, tick: int) -> Decimal:
        """Success a zone's crowding adds (negative: takes) for a trip set out now.

        Each trip short of an even share of the window's trips adds ``per_trip``,
        each trip over it takes as much, within ``cap``. A zone everyone uses
        gets harder and the ones nobody uses easier, so trips spread out.
        """
        settings = self._config.crowding
        if settings is None:
            return ZERO
        counts = self.traffic(tick)
        total = sum(counts.values())
        if not total:
            return ZERO
        share = Decimal(total) / Decimal(len(counts))
        modifier = settings.per_trip * (share - Decimal(counts[zone_id]))
        modifier = max(-settings.cap, min(settings.cap, modifier))
        return modifier.quantize(Decimal("0.0001"))

    # -- gear --------------------------------------------------------------
    def gear_of(self, account: AgentAccount) -> dict[Product, Decimal]:
        """What this adventurer's subclass can carry, and what each item adds."""
        subclass = self._subclass_of(account)
        return dict(self._config.gear.get(subclass, {})) if subclass is not None else {}

    def to_equip(self, account: AgentAccount) -> list[Product]:
        """Durable gear of its subclass the adventurer holds but does not wear.

        Setting out equips each of these.
        """
        return [
            product
            for product in sorted(self.gear_of(account), key=lambda p: p.value)
            if product in self._durable
            and account.equipment.get(product, 0) <= 0
            and account.available_quantity(product) > 0
        ]

    def supplies_for(
        self, account: AgentAccount, multiplier: Callable[[Product], Decimal]
    ) -> dict[Product, int]:
        """Units of each consumable (potion) in this subclass's gear one trip uses up.

        One unit each, times the demand_surge factor for that product; a
        fractional result rounds up, so a surge never rounds down to nothing.
        """
        return {
            product: int((Decimal(1) * multiplier(product)).to_integral_value(ROUND_CEILING))
            for product in sorted(self.gear_of(account), key=lambda p: p.value)
            if product not in self._durable
        }

    def gear_bonus(
        self, account: AgentAccount, supplied: dict[Product, Decimal] | None = None
    ) -> dict[Product, Decimal]:
        """The success probability each carried item adds.

        An equipped item counts while it has durability left, and so does one
        held unworn, which setting out equips (``to_equip``); a potion counts in
        proportion to how much of what the trip needed was carried (``supplied``,
        a fraction per potion). Items outside the subclass's gear add nothing.
        """
        bonus: dict[Product, Decimal] = {}
        for product, value in sorted(self.gear_of(account).items(), key=lambda kv: kv[0].value):
            if product in self._durable:
                if account.equipment.get(product, 0) > 0 or account.available_quantity(product) > 0:
                    bonus[product] = value
            elif supplied and supplied.get(product, ZERO) > ZERO:
                bonus[product] = value * supplied[product]
        return bonus

    def odds_now(
        self, account: AgentAccount, supplies: dict[Product, int], tick: int | None = None
    ) -> dict[str, Decimal]:
        """Success probability in each zone if the adventurer set out now.

        The equipment worn counts, and each potion in the share of what the trip
        uses that the adventurer holds unreserved, as ``start`` would count it.
        With ``tick``, each zone's crowding counts too.
        """
        supplied = {
            product: min(Decimal(account.available_quantity(product)), Decimal(needed))
            / Decimal(needed)
            for product, needed in supplies.items()
            if needed
        }
        bonus = self.gear_bonus(account, supplied)
        return {
            zone.zone_id: self.success_probability(
                account,
                zone.risk,
                gear_bonus=bonus,
                crowding=ZERO if tick is None else self.crowding(zone.zone_id, tick),
            )
            for zone in self._zones.zones()
        }

    # -- starting ----------------------------------------------------------
    def start(
        self,
        account: AgentAccount,
        zone_id: str,
        tick: int,
        supplies: dict[Product, int] | None = None,
    ) -> tuple[ExpeditionState, dict[Product, int]]:
        """Send an adventurer to a zone, using up the potions it carries.

        ``check_start`` must pass first; the engine equips ``to_equip`` between
        the two.

        Whatever the adventurer holds of each potion is consumed, up to what the
        trip uses; carrying fewer gives a proportional share of that potion's
        bonus. The bonus of every item is fixed here, at departure (design v1.0
        s5.3). Returns the state and what was consumed.
        """
        self.check_start(account, zone_id, tick)
        zone = self._zones.zone(zone_id)
        consumed: dict[Product, int] = {}
        supplied: dict[Product, Decimal] = {}
        for product, needed in (supplies or {}).items():
            taken = account.consume(product, needed)
            if taken:
                consumed[product] = taken
            supplied[product] = Decimal(taken) / Decimal(needed) if needed else ZERO

        duration = self._config.duration_ticks[zone.risk]
        state = ExpeditionState(
            zone_id=zone.zone_id,
            risk=zone.risk,
            started_tick=tick,
            resolves_at_tick=tick + duration,
            gear_bonus=self.gear_bonus(account, supplied),
            # The others' trips, before this one counts.
            crowding=self.crowding(zone.zone_id, tick),
        )
        self._starts.append((tick, zone.zone_id))
        account.expedition = state
        return state, consumed

    def check_start(self, account: AgentAccount, zone_id: str, tick: int) -> None:
        """Raise unless the adventurer may set out for this zone now."""
        if account.role is not Role.ADVENTURER:
            raise ExpeditionRejected(f"{account.agent_id} is a {account.role}, not an adventurer")
        if account.expedition is not None:
            raise ExpeditionRejected(f"{account.agent_id} is already away on an expedition")
        if account.is_resting(tick):
            raise ExpeditionRejected(
                f"{account.agent_id} is resting after an expedition and cannot set out "
                f"before tick {account.recovering_until_tick}"
            )
        self._zones.zone(zone_id)

    # -- probability -------------------------------------------------------
    def success_probability(
        self,
        account: AgentAccount,
        risk: RiskLevel,
        *,
        gear_bonus: dict[Product, Decimal] | None = None,
        crowding: Decimal = ZERO,
    ) -> Decimal:
        """Probability of success for this adventurer in this difficulty.

        Zone difficulty sets the base; the subclass's advantage and each item of
        its gear add to it (design v1.0 s5.3, s4.1), and so does the zone's
        ``crowding``. ``gear_bonus`` is what was fixed at departure; without it,
        the equipment worn now counts and no potion does. Clamped so that nothing
        is ever certain in either direction.
        """
        probability = self._config.success_probability[risk]

        subclass = self._subclass_of(account)
        if subclass is not None:
            settings = self._config.subclasses.get(subclass)
            if settings is not None:
                probability += settings.success_bonus.get(risk, ZERO)

        bonus = gear_bonus if gear_bonus is not None else self.gear_bonus(account)
        probability += sum(bonus.values(), ZERO)
        probability += crowding

        return min(
            max(probability, self._config.min_success_probability),
            self._config.max_success_probability,
        )

    def expected_take(
        self,
        account: AgentAccount,
        zone: Zone,
        bonus: dict[Product, Decimal],
        price: Callable[[Resource], Decimal],
        crowding: Decimal = ZERO,
        units: Decimal | None = None,
        gold: Decimal | None = None,
    ) -> Decimal:
        """What one trip to ``zone`` is expected to bring back, valued in gold.

        Carrying the items in ``bonus`` (as ``gear_bonus`` gives them): the chance
        of success, the units a success or a failure yields valued at ``price``
        in the zone's mix (spawn rates scaled by the subclass's affinity), and a
        success's gold with each item's gold bonus. The configured spawn rates,
        not a shock's: agents learn of a shock from the market, as of any other.
        """
        probability = self.success_probability(
            account, zone.risk, gear_bonus=bonus, crowding=crowding
        )
        subclass = self._subclass_of(account)
        settings = self._config.subclasses.get(subclass) if subclass is not None else None
        affinity = settings.resource_affinity if settings is not None else {}
        weights = {r: rate * affinity.get(r, ONE) for r, rate in zone.spawn_per_tick.items()}
        total = sum(weights.values(), ZERO)
        unit = sum((price(r) * w for r, w in weights.items()), ZERO) / total if total else ZERO
        if units is None:
            units = Decimal(self._config.expected_yield[zone.risk])
        partial = units * self._config.partial_yield_on_failure
        if gold is None:
            gold = self._config.gold_loot.get(zone.risk, ZERO)
            if settings is not None:
                gold *= settings.loot_multiplier
            gold *= ONE + sum(
                (self._config.gear_gold_bonus.get(p, ZERO) for p, b in bonus.items() if b > ZERO),
                ZERO,
            )
        return probability * (units * unit + gold) + (ONE - probability) * partial * unit

    def zones(self) -> list[Zone]:
        return self._zones.zones()

    @staticmethod
    def _subclass_of(account: AgentAccount) -> AdventurerSubclass | None:
        try:
            return AdventurerSubclass(account.specialization)
        except ValueError:
            return None

    # -- resolution --------------------------------------------------------
    def resolve(self, account: AgentAccount, rng: random.Random, tick: int) -> ExpeditionResult:
        """Resolve one due expedition and deliver its outcome to the adventurer."""
        return self.resolve_group([(account, rng)], tick)[0]

    def resolve_group(
        self, due: list[tuple[AgentAccount, random.Random]], tick: int
    ) -> list[ExpeditionResult]:
        """Resolve the expeditions due in one zone at one tick, sharing its pool.

        Each adventurer's success and units are rolled first; then the units are
        drawn one at a time, taking turns in the given order, so that a pool too
        small for everyone is split about evenly (15 units among four asking 5
        each come back as 4, 4, 4 and 3) rather than emptied by whoever resolves
        first. Each draw uses the adventurer's own stream.
        """
        rolls = [(account, rng, *self._roll(account, rng, tick)) for account, rng in due]
        # Gold the same way: each success is due its full gold; a zone whose
        # gold pool cannot cover them all pays each the same share of it.
        dues = [
            self._gold_due(account) if succeeded else ZERO
            for account, _rng, succeeded, *_rest in rolls
        ]
        paid = dues
        if rolls:
            zone = self._zones.zone(rolls[0][0].expedition.zone_id)
            owed = sum(dues, ZERO)
            if zone.gold_rate is not None and owed > zone.gold:
                share = zone.gold / owed
                paid = [(d * share).quantize(Decimal("0.01"), rounding=ROUND_DOWN) for d in dues]
            zone.take_gold(sum(paid, ZERO))
        yielded: list[dict[Resource, int]] = [{} for _ in rolls]
        left = [units for *_, units in rolls]
        while any(left):
            for i, (account, rng, *_rest) in enumerate(rolls):
                if not left[i]:
                    continue
                zone = self._zones.zone(account.expedition.zone_id)
                resource = self._draw_one(account, zone, rng)
                if resource is None:
                    left = [0] * len(left)  # the zone is exhausted for everyone
                    break
                yielded[i][resource] = yielded[i].get(resource, 0) + 1
                left[i] -= 1
        return [
            self._deliver(account, succeeded, probability, recovering_until, got, gold, due)
            for (
                account,
                _rng,
                succeeded,
                probability,
                recovering_until,
                _units,
            ), got, gold, due in zip(rolls, yielded, paid, dues, strict=True)
        ]

    def _gold_due(self, account: AgentAccount) -> Decimal:
        """A success's full gold: the zone's loot, each carried item's share, the subclass's."""
        state = account.expedition
        assert state is not None
        gold = self._config.gold_loot.get(state.risk, ZERO)
        # Each item carried adds its share of the trip's gold (fixed at departure,
        # with the success bonus): equipment pays in what a success brings back.
        gold *= ONE + sum(
            (
                self._config.gear_gold_bonus.get(product, ZERO)
                for product, bonus in state.gear_bonus.items()
                if bonus > ZERO
            ),
            ZERO,
        )
        subclass = self._subclass_of(account)
        if subclass is not None and subclass in self._config.subclasses:
            gold *= self._config.subclasses[subclass].loot_multiplier
        return gold

    def _roll(
        self, account: AgentAccount, rng: random.Random, tick: int
    ) -> tuple[bool, Decimal, int, int]:
        """Success, its probability, the end of the rest, and the units due."""
        state = account.expedition
        if state is None:  # pragma: no cover -- callers check before resolving
            raise ExpeditionRejected(f"{account.agent_id} is not on an expedition")

        probability = self.success_probability(
            account, state.risk, gear_bonus=state.gear_bonus, crowding=state.crowding
        )
        succeeded = rng.randrange(PROBABILITY_SCALE) < int(probability * PROBABILITY_SCALE)

        expected = self._config.expected_yield[state.risk]
        if succeeded:
            spread = self._config.yield_spread[state.risk]
            units = max(0, expected + rng.randint(-spread, spread))
            recovering_until = tick + self._config.rest_ticks
        else:
            # A partial loss, not a total one: the fractional part of the
            # partial yield is drawn, so a small haul is not always truncated to 0.
            partial = Decimal(expected) * self._config.partial_yield_on_failure
            units = int(partial)
            if rng.randrange(PROBABILITY_SCALE) < int((partial - units) * PROBABILITY_SCALE):
                units += 1
            # A failure adds its risk's recovery on top of the ordinary rest.
            recovering_until = (
                tick + self._config.rest_ticks + self._config.recovery_ticks_on_failure[state.risk]
            )
        return succeeded, probability, recovering_until, units

    def _deliver(
        self,
        account: AgentAccount,
        succeeded: bool,
        probability: Decimal,
        recovering_until: int,
        yielded: dict[Resource, int],
        gold: Decimal,
        gold_due: Decimal,
    ) -> ExpeditionResult:
        """Hand over the haul and the gold, wear the gear, and end the trip."""
        state = account.expedition
        assert state is not None
        for resource, quantity in sorted(yielded.items(), key=lambda kv: kv[0].value):
            account.receive(resource, quantity)
        if gold > ZERO:
            account.credit_gold(gold)

        broken = self._wear_equipment(account)

        account.expedition = None
        account.recovering_until_tick = recovering_until

        return ExpeditionResult(
            agent_id=account.agent_id,
            zone_id=state.zone_id,
            risk=state.risk,
            outcome=ExpeditionOutcome.SUCCESS if succeeded else ExpeditionOutcome.FAILURE,
            success_probability=probability,
            yielded=yielded,
            recovering_until_tick=recovering_until,
            equipment_broken=broken,
            gold=gold,
            gold_due=gold_due,
        )

    def _draw_one(self, account: AgentAccount, zone: Zone, rng: random.Random) -> Resource | None:
        """Draw one unit from the zone's pool, or None when the pool is empty.

        A unit's resource is chosen by weight -- the zone's appearance rate,
        scaled by the subclass's affinity -- among resources the zone still has
        in stock. Drawing unit by unit rather than in bulk is what makes scarcity
        bite: once the pool of a resource empties, further units come back as
        something else, which is the supply reorganisation the primary scenario
        is designed to observe.
        """
        subclass = self._subclass_of(account)
        affinity = (
            self._config.subclasses[subclass].resource_affinity
            if subclass is not None and subclass in self._config.subclasses
            else {}
        )
        weights: list[tuple[Resource, int]] = []
        for resource in sorted(zone.spawn_per_tick, key=lambda r: r.value):
            if zone.stock(resource) <= 0:
                continue
            weight = zone.spawn_per_tick[resource] * affinity.get(resource, ONE)
            scaled = int(weight * WEIGHT_SCALE)
            if scaled > 0:
                weights.append((resource, scaled))
        if not weights:
            return None
        total = sum(weight for _, weight in weights)
        roll = rng.randrange(total)
        cumulative = 0
        for resource, weight in weights:
            cumulative += weight
            if roll < cumulative:
                return resource if zone.take(resource, 1) else None
        return None  # pragma: no cover -- roll < total always lands

    def _wear_equipment(self, account: AgentAccount) -> list[Product]:
        """Apply expedition wear; return whatever broke."""
        loss = self._config.durability_loss_per_expedition
        broken: list[Product] = []
        for product in sorted(account.equipment, key=lambda p: p.value):
            remaining = account.equipment[product] - loss
            if remaining <= 0:
                del account.equipment[product]
                broken.append(product)
            else:
                account.equipment[product] = remaining
        return broken
