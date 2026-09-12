"""What an activated agent is allowed to see (design v1.0 s5.2, s7).

An economic agent observes its own role, traits, gold, inventory, equipment and
history; its own orders and trades; best bid/ask and the last N trades of the
goods in its fixed interest list; the public directory; and its messages, with
the REQUESTs and OFFERs addressed to it that are still open (B and C). In
Condition C it additionally receives the passive bulletin.

It never sees another agent's gold, inventory or plans, and never any aggregate
of scarcity, production or unmet demand -- those belong to the Director's global
view in A and to the bulletin in C. Building the observation here, once, is what
keeps that boundary in a single reviewable place instead of spread across prompt
templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from rpg_economy.config.models import SimulationConfig, TraitsConfig
from rpg_economy.domain import AdventurerSubclass, Good, OrderSide, Product, Resource, Role
from rpg_economy.engine import Engine
from rpg_economy.engine.market import Order, Trade
from rpg_economy.engine.state import AgentAccount
from rpg_economy.messaging import Message
from rpg_economy.messaging.bus import OPENING_TYPES


@dataclass(frozen=True)
class MarketView:
    """The permitted market window for one good."""

    good: str
    best_bid: Decimal | None
    best_ask: Decimal | None
    recent_prices: list[Decimal]
    has_offer: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "good": self.good,
            "best_bid": None if self.best_bid is None else str(self.best_bid),
            "best_ask": None if self.best_ask is None else str(self.best_ask),
            "recent_prices": [str(p) for p in self.recent_prices],
            "has_offer": self.has_offer,
        }


@dataclass(frozen=True)
class Observation:
    """Everything one activation may look at."""

    agent_id: str
    role: str
    specialization: str | None
    tick: int
    cycle: int
    traits: TraitsConfig
    gold: Decimal
    available_gold: Decimal
    #: Wealth tax the agent owes and could not pay. A lien: it is collected as
    #: soon as gold is free, and available_gold already leaves it out.
    tax_owed: Decimal
    inventory: dict[str, int]
    #: The part of the inventory not already promised to the agent's own open
    #: sell orders: what it can still sell, equip or use.
    available_inventory: dict[str, int]
    equipment: dict[str, int]
    #: On an expedition right now.
    is_away: bool
    production_capacity: int
    open_orders: list[dict[str, Any]]
    market: list[MarketView]
    directory: list[dict[str, str | None]]
    messages: list[Message] = field(default_factory=list)
    #: The agent's own most recent trades (design s5.2: its own orders and
    #: trades). Each: tick, side, good, quantity, price.
    own_trades: list[dict[str, Any]] = field(default_factory=list)
    #: REQUESTs and OFFERs addressed to the agent that are still open and not
    #: among ``messages``: received in an earlier activation, not yet answered
    #: or expired. B and C only. Each: message_id, sender_id, type, good,
    #: quantity, price, ticks_left.
    open_proposals: list[dict[str, Any]] = field(default_factory=list)
    #: The inbox's REQUESTs and OFFERs that are still open -> their ticks_left
    #: (None without an expiry). One in the inbox but not here has expired, been
    #: replaced by a newer one from its sender about its good, or been answered:
    #: it is shown as information and can no longer be answered.
    inbox_open: dict[int, int | None] = field(default_factory=dict)
    bulletin: dict[str, Any] | None = None
    #: Back from an expedition but resting: the first tick it can set out again.
    can_set_out_from_tick: int | None = None
    #: The products that raise this adventurer's expedition success (s5.3).
    #: Gear this adventurer's subclass can carry -> success probability it adds.
    expedition_gear: dict[str, str] = field(default_factory=dict)
    #: An adventurer's success probability per zone if it set out now, with the
    #: equipment it wears and the potions it holds. Empty for artisans.
    success_chance: dict[str, str] = field(default_factory=dict)
    #: Plain-language facts about the goods the agent follows, from the book's
    #: best bid and ask and the agent's own expired orders, plus for an artisan
    #: what each of its recipes could make now and whether a buyer is waiting.
    signals: list[str] = field(default_factory=list)
    #: Headline lines, shown above the state. An artisan's: products buyers
    #: want (to sell, to craft, or the inputs it lacks for them), stock that
    #: does not sell, inputs its bids failed to buy. An adventurer's: raw
    #: materials it holds that buyers bid for.
    status: list[str] = field(default_factory=list)
    #: An adventurer's gear it lacks, each with what it costs and adds a trip
    #: (see ``gear_facts``); empty for artisans.
    gear: list[dict[str, Any]] = field(default_factory=list)
    #: The agent's own recent actions and what the engine did with them.
    history: list[dict[str, Any]] = field(default_factory=list)

    def message_payloads(self) -> list[dict[str, Any]]:
        """The inbox as the agent sees it: each REQUEST and OFFER marked open or not.

        An open one carries its ticks_left; a closed one (expired, replaced or
        answered) is kept, as information, with ``open`` false and no ticks_left.
        """
        payloads = []
        for message in self.messages:
            payload = message.to_payload()
            if message.type in OPENING_TYPES:
                payload["open"] = message.message_id in self.inbox_open
                payload["ticks_left"] = self.inbox_open.get(message.message_id)
            payloads.append(payload)
        return payloads

    def to_payload(self) -> dict[str, Any]:
        """The delivered observation, in full, for the run log (s15, s4.12).

        Logged complete rather than hashed. A hash lets a reader verify a replay
        they already have; it does not let them perform one. Section 15 asks for
        reconstruction, so the log has to carry what the agent actually saw,
        including the bulletin and inbox, which are what distinguish the
        conditions.
        """
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "specialization": self.specialization,
            "tick": self.tick,
            "cycle": self.cycle,
            "traits": {
                "risk_tolerance": str(self.traits.risk_tolerance),
                "time_orientation": str(self.traits.time_orientation),
                "commercial_aggressiveness": str(self.traits.commercial_aggressiveness),
                "adaptability": str(self.traits.adaptability),
            },
            "gold": str(self.gold),
            "available_gold": str(self.available_gold),
            "tax_owed": str(self.tax_owed),
            "inventory": dict(self.inventory),
            "available_inventory": dict(self.available_inventory),
            "equipment": dict(self.equipment),
            "expedition_gear": dict(self.expedition_gear),
            "success_chance": dict(self.success_chance),
            "signals": list(self.signals),
            "status": list(self.status),
            "gear": [dict(fact) for fact in self.gear],
            "is_away": self.is_away,
            "can_set_out_from_tick": self.can_set_out_from_tick,
            "production_capacity": self.production_capacity,
            "open_orders": [dict(order) for order in self.open_orders],
            "own_trades": [dict(trade) for trade in self.own_trades],
            "market": [view.to_payload() for view in self.market],
            "directory": [dict(entry) for entry in self.directory],
            "messages": self.message_payloads(),
            "open_proposals": [dict(proposal) for proposal in self.open_proposals],
            "bulletin": self.bulletin,
            "history": [dict(entry) for entry in self.history],
        }


ZERO_GOLD = Decimal(0)
CENT = Decimal("0.01")

#: The order of an artisan's headline lines, most pressing first.
HEADLINE_ORDER = ("Ready to sell", "Wanted", "Not offered", "Unsold", "Buying", "Too dear")
#: Units of a product held unsold, with no buyer, from which crafting more is flagged.
UNSOLD_FLAG = 3


def stepped(
    config: SimulationConfig,
    price: Decimal,
    up: bool,
    engine: Engine | None = None,
    good: Good | None = None,
) -> str:
    """The direction of a price signal, with the step it suggests when one is set.

    Told only "ask less" or "bid above", agents moved 0.05 at a time,
    so a price took most of a phase to answer a change in supply.
    """
    step = config.market.price_step_pct
    if step is None:
        return f"above {price}" if up else "less"
    if engine is not None and good is not None:
        return f"about {moved(config, engine, good, price, up)}"
    factor = 1 + step if up else 1 - step
    return f"about {(price * factor).quantize(CENT)}"


def moved(
    config: SimulationConfig, engine: Engine, good: Good, price: Decimal, up: bool
) -> Decimal:
    """``price`` one step up or down, a larger one back toward a raw material's typical price.

    Stepped 10% at a time, herbs took 70 ticks of the recovery to
    come from 14 to 10 with sellers offering, against a 6.00 typical price.
    A raw material moving back toward its typical price goes halfway there
    when that is more than a step: down when it is dear and offered, up when
    it is cheap and short. Never past the typical price on the way back.
    """
    step = config.market.price_step_pct or Decimal(0)
    out = price * (1 + step) if up else price * (1 - step)
    reference, _ = reference_price(config, engine, good)
    if isinstance(good, Resource) and reference is not None:
        half = (price + reference) / 2
        if up and price < reference:
            out = max(out, half)
        elif not up and price > reference:
            out = max(reference, min(out, half))
    return out.quantize(CENT)


def recent_sale(config: SimulationConfig, engine: Engine, good: Good) -> Trade | None:
    """The good's last trade when it was within ``recent_sale_ticks``, else None."""
    window = config.market.recent_sale_ticks
    last = engine.market.last_trades(good, 1)
    if window is None or not last or last[-1].tick < engine.tick - window:
        return None
    return last[-1]


def rebid(config: SimulationConfig, engine: Engine, good: Good, top: Decimal) -> str:
    """What a buyer whose bids up to ``top`` lapsed unfilled, with nothing offered, should bid.

    Adventurers bring raw materials back in bursts, so a bid often
    lapses with no seller in town; stepped up each time, iron rose 46% with
    more gathered than used. Only a good that stopped selling is bid up.
    """
    sale = recent_sale(config, engine, good)
    if sale is None:
        return f"bid {stepped(config, top, True, engine, good)} to get any"
    ago = engine.tick - sale.tick
    return (
        f"it sold at {sale.price} {ago} tick(s) ago and sellers come back between trips: "
        f"bid about {max(sale.price, top)}"
    )


def toward_typical(
    config: SimulationConfig, engine: Engine, account: AgentAccount, good: Good, price: Decimal
) -> tuple[Decimal, Decimal | None]:
    """A raw material's ``price`` a step closer to its typical price, and that price.

    After the herb shock alchemists were told to bid the last sale
    while herbs kept selling, and herbs held at 11.50 to 12.87, twice their
    6.00 typical price, for 100 ticks with 17 left unsold. A buyer whose bids
    fill while the material sells above its typical price steps down; one
    whose bids lapse is short of it and does not, and once it stops selling
    the step up returns. Products follow their inputs instead.
    """
    reference, _ = reference_price(config, engine, good)
    step = config.market.price_step_pct
    if (
        not isinstance(good, Resource)
        or account.role is not Role.ARTISAN
        or reference is None
        or step is None
        or price <= reference
        or _lapsing(config, engine, account, good)
    ):
        return price, reference
    return moved(config, engine, good, price, False), reference


def reference_price(
    config: SimulationConfig, engine: Engine, good: Good
) -> tuple[Decimal | None, bool]:
    """A good's typical price over the window, or its guide value below enough trades.

    Returns the price and whether it is the guide's. None when no window is set.
    """
    window = config.market.usual_price_ticks
    if not window:
        return None, False
    since = engine.tick - window
    if engine.market.trade_count(good, since) < config.market.usual_price_min_trades:
        guide = Decimal(str(config.economy.initial_reference_price[good.value]))
        return guide.quantize(CENT), True
    return engine.market.median_price(good, since), False


def _halfway(config: SimulationConfig, good: Good, bid: Decimal, ask: Decimal) -> Decimal | None:
    """The midpoint of a narrow raw-material spread, or None.

    Alchemists bid 6.00 for herbs and adventurers asked 6.60, both
    re-posting the same price as their orders lapsed, and no herb traded for 30
    ticks. Only raw materials: an adventurer told a lower bid may do on gear
    could set out without it.
    """
    step = config.market.price_step_pct
    if step is None or not isinstance(good, Resource) or ask <= bid:
        return None
    if ask > bid * (1 + 2 * step):
        return None
    return ((bid + ask) / 2).quantize(CENT)


def _lapsing(config: SimulationConfig, engine: Engine, account: AgentAccount, good: Good) -> bool:
    """Whether the agent's bids for ``good`` lapsed unfilled lately with none bought since."""
    market = engine.market
    since = engine.tick - 2 * config.market.order_expiry_ticks
    lapsed = [
        o.expires_at_tick
        for o in market.expired_unfilled_of(account.agent_id, since)
        if o.good == good and o.side is OrderSide.BUY
    ]
    bought = [
        t.tick
        for t in market.trades_of(account.agent_id, 50)
        if t.good == good and t.buyer_id == account.agent_id
    ]
    return bool(lapsed) and (not bought or max(bought) < max(lapsed))


def _unfilled_bids(
    config: SimulationConfig, engine: Engine, account: AgentAccount, good: Good
) -> list[Order]:
    """The artisan's bids for ``good`` that expired unfilled since its last sale, newest first."""
    market = engine.market
    since = engine.tick - 2 * config.market.order_expiry_ticks
    last = market.last_trades(good, 1)
    if last:
        since = max(since, last[-1].tick)
    return [
        o
        for o in market.expired_unfilled_of(account.agent_id, since)
        if o.good == good and o.side is OrderSide.BUY
    ]


def _lowball(
    config: SimulationConfig, engine: Engine, good: Good, bid: Decimal, cost: Decimal
) -> str:
    """Why a bid at or under an artisan's input cost is no reason to sell.

    Told only that a sale at cost "earns nothing", alchemists still
    sold every potion at cost "to reduce storage" and "clear inventory": the
    prompt warned that holding costs storage but never said how little.
    """
    text = (
        "below what one costs you: selling there loses gold"
        if bid < cost
        else "equal to what one costs you: selling there earns nothing"
    )
    reference, _ = reference_price(config, engine, good)
    storage = config.economy.storage_cost_per_unit.get(Role.ARTISAN)
    if reference is not None and reference > cost and storage is not None:
        text += (
            f"; holding one costs {storage.normalize():f} a tick, so waiting for a buyer "
            f"near its {reference} reference costs little"
        )
    return text


def _ask_down(
    config: SimulationConfig, engine: Engine, good: Good, ask: Decimal, cost: Decimal | None
) -> str:
    """A seller's step down, never below an artisan's input cost."""
    text = stepped(config, ask, False, engine, good)
    step = config.market.price_step_pct
    if cost is not None and step is not None and moved(config, engine, good, ask, False) < cost:
        return f"less, but not below its {cost} cost"
    return text


def confiscation_lines(engine: Engine, account: AgentAccount) -> list[str]:
    """What the shock took from the agent and paid for it, while the shock lasts."""
    shock = engine.shocks.active
    if shock is None or account.agent_id not in shock.confiscated:
        return []
    quantity, paid = shock.confiscated[account.agent_id]
    good = shock.details["target_resource"]
    line = (
        f"Confiscated: at tick {shock.applied_tick} the government took {quantity} "
        f"{good} of yours and paid you {paid.quantize(CENT)} gold for them, "
        f"{(paid / quantity).quantize(CENT)} each"
    )
    if account.agent_id in shock.orders_cut:
        before, after = shock.orders_cut[account.agent_id]
        line += (
            f"; your sell orders for {before} were cut to {after} at the same prices"
            if after
            else f"; your sell orders for {before} were withdrawn"
        )
    return [line]


def adventurer_status(config: SimulationConfig, engine: Engine, account: AgentAccount) -> list[str]:
    """Headline lines for an adventurer: the raw materials it holds that buyers bid for.

    In the signals alone, adventurers holding raw materials a buyer waited for
    offered them 18-25% of the time in real runs, and traded potions
    instead. Stated as facts, and saying that the only seller sets the price:
    shown only the bid, sellers took it and prices stopped moving (2.23, 2.32).
    """
    market = engine.market
    lines: list[str] = []
    for resource in Resource:
        held = account.available_quantity(resource)
        bid = market.best_bid(resource)
        if not held or bid is None:
            continue
        ask = market.best_ask(resource)
        # 2.33: told "buyers bid 6.00 and nobody else offers it", sellers took
        # the bid and the scarce good's price
        # fell during its shock; as the only seller it sets the price.
        lines.append(
            f"Wanted: {resource.value}: you hold {held} and nobody else offers it; buyers "
            f"bid {bid} so far, and you set the price"
            if ask is None
            else f"Wanted: {resource.value}: you hold {held}; buyers bid {bid}, others ask {ask}"
        )
    return (
        _main_item_lines(config, engine, account)
        + sorted(lines)
        + zone_lines(config, engine, account)
        + gear_lines(gear_facts(config, engine, account))
    )


def _main_item_lines(config: SimulationConfig, engine: Engine, account: AgentAccount) -> list[str]:
    """A headline when the adventurer's main item is gone and it carries none.

    Both warriors' weapons broke, and told in the setting-out line to
    bid about 48 for one, they set out without one in 10 answers of 20.
    """
    try:
        subclass = AdventurerSubclass(account.specialization)
    except ValueError:
        return []
    main = config.expeditions.starting_gear.get(subclass)
    if main is None:
        return []
    if account.equipment.get(main, 0) > 0 or account.available_quantity(main) > 0:
        return []
    mine = engine.market.open_orders_of(account.agent_id)
    if any(o.good == main and o.side is OrderSide.BUY for o in mine):
        return []
    fact = {f["item"]: f for f in gear_facts(config, engine, account)}.get(main.value)
    if fact is None:
        return []
    gains = [Decimal(g) for g in fact["gains"].values()]
    if fact["ask"] is not None:
        where, action = f"offered at {fact['ask']}", f"buy one at {fact['ask']}"
    else:
        where, action = "nobody offers one yet", f"bid about {fact['price']} for one"
    lasts = f" and lasts {fact['durability']} trips" if fact["durability"] else ""
    return [
        f"Broken: your {main.value} broke and you carry none ({where}): {action} before "
        f"you set out; it adds about {min(gains):.0f}-{max(gains):.0f} a trip{lasts}"
    ]


def zone_lines(config: SimulationConfig, engine: Engine, account: AgentAccount) -> list[str]:
    """How crowded each zone has been, what that does to the odds, and what a trip is worth.

    LLM adventurers went to the safe zone on 22 trips in 23, though a
    trip to the hard one was worth about twice as much a tick; the prompt gave
    the parts (odds, haul, gold, duration) but never the sum. Who set out where
    is public, like adventurers seen leaving town; what is left in a zone is not
    shown, and a crowded zone's trips bring back less of it.
    """
    settings = config.expeditions.crowding
    if settings is None:
        return []
    expeditions = engine.expeditions
    market = engine.market
    needed = expeditions.supplies_for(account, lambda product: Decimal(1))
    supplied = {
        product: min(Decimal(account.available_quantity(product)), Decimal(units)) / units
        for product, units in needed.items()
    }
    carried = expeditions.gear_bonus(account, supplied)

    def price(good: Good) -> Decimal:
        last = market.last_trades(good, 1)
        if last:
            return last[-1].price
        return Decimal(str(config.economy.initial_reference_price[good.value]))

    traffic = engine.zone_traffic()
    crowding = engine.zone_crowding()
    parts = []
    for zone in expeditions.zones():
        shift = crowding[zone.zone_id]
        # Valued at the full haul, the hard zone drew 41 trips of
        # 60, though its pool held about 4 of the 7 units; the likely haul
        # shares what the zone grows among the trips that went there.
        haul = expeditions.likely_haul(zone.zone_id, engine.tick)
        # A crowded zone gave 1 unit in three trips and still 194
        # gold, so the crowd cost nothing; its gold is shared too, and said.
        full = expeditions.full_gold(account, zone, carried)
        gold = expeditions.likely_gold(zone.zone_id, engine.tick, full)
        worth = expeditions.expected_take(account, zone, carried, price, shift, haul, gold)
        ticks = config.expeditions.duration_ticks[zone.risk] + config.expeditions.rest_ticks
        state = "crowded" if shift < 0 else ("quiet" if shift > 0 else "even")
        trips = traffic[zone.zone_id]
        whole = full.quantize(Decimal(1))
        cap = zone.gold_cap.quantize(Decimal(1)) if zone.gold_cap is not None else None
        if gold >= whole:
            thinned = f"its full {whole} gold"
        elif cap is not None and gold >= cap:
            thinned = f"{gold} gold of a full {whole}, all the zone holds when no one has been"
        else:
            thinned = f"{gold} gold of a full {whole}, thinned by the trips there"
        parts.append(
            f"{zone.zone_id}: {trips} trip{'' if trips == 1 else 's'} set out in the last "
            f"{settings.window_ticks} ticks ({state}, success {shift:+.2f}), a success "
            f"likely finds about {haul.normalize():f} units and {thinned}, a trip worth "
            f"about {worth.quantize(Decimal(1))} to you, {(worth / ticks).quantize(CENT)} a "
            "tick with the rest after it"
        )
    return ["Zones now: " + "; ".join(parts)]


def gear_facts(
    config: SimulationConfig, engine: Engine, account: AgentAccount
) -> list[dict[str, Any]]:
    """What each item an adventurer lacks would add to a trip, against its cost.

    Gear is optional. With success alone, no item paid for itself in the zone
    LLM adventurers favour, and they bought none; with gear also adding to a
    success's gold it pays, and these facts say by how much, from the same
    rules the engine applies (``ExpeditionSystem.expected_take``) at market
    prices, so the choice is the adventurer's to make on facts. An item is
    listed when it is not carried: not worn or held (setting out equips a held
    item), or, a potion, not held. Each: item, ask (None when nobody offers one), price (the ask,
    else its last trade, else the guide), traded (it has traded), durability,
    cost_per_trip, gains (zone -> what it adds to a trip) and pays (it adds more
    than it costs a trip in every zone).
    """
    expeditions = engine.expeditions
    gear = expeditions.gear_of(account)
    if not gear:
        return []
    market = engine.market
    needed = expeditions.supplies_for(account, lambda product: Decimal(1))
    supplied = {
        product: min(Decimal(account.available_quantity(product)), Decimal(units)) / units
        for product, units in needed.items()
    }
    carried = expeditions.gear_bonus(account, supplied)

    def price(good: Good) -> Decimal:
        last = market.last_trades(good, 1)
        if last:
            return last[-1].price
        return Decimal(str(config.economy.initial_reference_price[good.value]))

    def take(zone, bonus: dict[Product, Decimal]) -> Decimal:
        # At the haul and gold a success there is likely to find: with the
        # zones' gold pooled, an item's gold share is worth only what the zone
        # still holds.
        tick = engine.tick
        shift = expeditions.crowding(zone.zone_id, tick)
        haul = expeditions.likely_haul(zone.zone_id, tick)
        gold = expeditions.likely_gold(
            zone.zone_id, tick, expeditions.full_gold(account, zone, bonus)
        )
        return expeditions.expected_take(account, zone, bonus, price, shift, haul, gold)

    def gains_of(with_item: dict[Product, Decimal], without: dict[Product, Decimal]) -> dict:
        return {
            zone.zone_id: str((take(zone, with_item) - take(zone, without)).quantize(Decimal(1)))
            for zone in expeditions.zones()
        }

    facts = []
    for product, bonus in sorted(gear.items(), key=lambda kv: kv[0].value):
        if carried.get(product, ZERO_GOLD) > ZERO_GOLD:
            continue
        gains = gains_of({**carried, product: bonus}, carried)
        durability = config.economy.durability.get(product)
        ask = market.best_ask(product)
        cost = ask if ask is not None else price(product)
        per_trip = (cost / durability if durability else cost).quantize(CENT)
        facts.append(
            {
                "item": product.value,
                "ask": None if ask is None else str(ask),
                "price": str(cost.quantize(CENT)),
                "traded": bool(market.last_trades(product, 1)),
                "durability": durability,
                "cost_per_trip": str(per_trip),
                "gains": gains,
                "pays": min(Decimal(g) for g in gains.values()) > per_trip,
            }
        )
    return facts


def gear_lines(facts: list[dict[str, Any]]) -> list[str]:
    """The adventurer's "Gear" headlines, one per item it lacks."""
    lines = []
    for fact in facts:
        adds = ", ".join(f"{gain} in {zone}" for zone, gain in fact["gains"].items())
        terms = (
            f"lasts {fact['durability']} trips, {fact['cost_per_trip']} a trip"
            if fact["durability"]
            else "used up on the trip"
        )
        sold = "it last sold for" if fact["traded"] else "guide price"
        where = (
            f"offered at {fact['ask']}"
            if fact["ask"] is not None
            else f"nobody offers one; {sold} {fact['price']}"
        )
        lines.append(f"Gear: {fact['item']} ({where}; {terms}) adds about {adds} a trip")
    return lines
