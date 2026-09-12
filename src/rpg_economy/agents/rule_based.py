"""Rule-based agents: simple deterministic heuristics per role.

These exercise the economy end to end without any LLM, which is
what makes the determinism gate meaningful and what lets the whole pipeline run
offline. They are not a baseline condition and are not part of the
experiment: conditions A, B and C all use LLM agents.

Each policy emits exactly one ``AgentAction`` per activation, matching the design's
one-decision-per-activation structure (s7). Every draw comes from the agent's own
seeded stream, so an agent's behaviour depends on the run seed and its own
history alone -- never on how many other agents acted first.
"""

from __future__ import annotations

import random
from decimal import Decimal

from rpg_economy.agents.action import PRICE_QUANTUM, AgentAction, wait
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import (
    ActionType,
    AdventurerSubclass,
    Good,
    OrderSide,
    Priority,
    Product,
    Resource,
    Role,
)
from rpg_economy.engine import Engine
from rpg_economy.engine.state import AgentAccount

#: What a seller asks over the recent price when it is *not* under pressure.
#: Zero, like the buyer's premium: if either side quotes off the recent price by
#: default, every trade lands on that side of it, the recent price moves with
#: each trade, and the next quote compounds on it -- the price drifts for ever
#: whatever supply and demand are doing. Quoting the recent price and moving
#: only under pressure (a blocked buyer raises, an unsold seller concedes) lets
#: it stand still in a balanced market and move when one side is short.
SELLER_MARGIN = Decimal("0")
#: What a seller cuts per threshold's worth of stock held above it. Excess
#: stock is what brings a price back down once a shortage ends.
OVERSTOCK_STEP = Decimal("0.03")
#: The most a seller will cut, however much stock it is sitting on.
MAX_SELLER_DISCOUNT = Decimal("0.10")
#: The most a seller raises its price over its belief when its offers keep selling.
MAX_SELLER_PREMIUM = Decimal("0.20")
#: A buyer in no hurry bids a little under the recent price and raises it one
#: escalation step per activation it goes unfilled. The small bargain is the
#: downward force that brings a price back once supply returns; the escalation
#: is the upward one under scarcity. The anchored reference keeps a balanced
#: market from drifting either way.
BUYER_PREMIUM = Decimal("-0.02")
#: An adventurer sells a resource once it holds more than this.
ADVENTURER_STOCK_THRESHOLD = 0
#: An adventurer holds a resource back while its price is below this fraction
#: of the starting one, for up to HOLD_PATIENCE ticks, in case it recovers. It
#: has no use for raw materials, so after that it sells at what the market pays.
HOLD_FRACTION = Decimal("0.6")
HOLD_PATIENCE = 15
#: Bids for a piece of gear an adventurer places before setting out without it,
#: and the ticks before it tries again.
GEAR_PATIENCE = 3
GEAR_RETRY_TICKS = 15
#: An adventurer restocks a consumable when it holds fewer than this.
CONSUMABLE_RESTOCK_THRESHOLD = 1
#: How many units of a consumable an adventurer buys at a time.
CONSUMABLE_RESTOCK_QUANTITY = 2
#: The most potions of one kind an adventurer learns to carry for a trip.
MAX_POTIONS_PER_TRIP = 4
#: Largest lot any agent offers in a single order.
#:
#: Dumping a whole position in one order produces one transaction where the
#: same quantity, worked in lots, produces several. That matters beyond
#: tidiness: the price series is built from transactions, so a market that
#: clears in a handful of large blocks gives too few observations per phase to
#: measure volatility or a recovery path at all.
MAX_LOT = 4


#: Weight of a good's starting price in an agent's belief of what it is worth.
#: An agent that trusts only the last trades has no sense of value: each
#: concession lowers the next reference, and a market with a slight surplus
#: slides without end. Anchoring part of the belief gives prices a level to
#: return to, so the base phase is stationary and a shock shows as a move
#: away from it and back.
REFERENCE_ANCHOR = Decimal("0.3")
#: The most a buyer bids over a good's starting price, however long it has gone
#: without. Escalation is measured from the recent price, and under a lasting
#: shortage every trade lifts the recent price, so without a ceiling each round
#: of escalation starts higher than the last and the price spirals without end.
#: The ceiling leaves room for a shock to show in the price and stops the spiral.
MAX_BID_OVER_START = Decimal("0.6")


def reference_price(engine: Engine, good: Good) -> Decimal:
    """What the agent believes ``good`` is worth: recent trades, anchored.

    Deliberately derived from trades rather than from any stored quote: the
    design forbids an official price (s5.2), so this is a *belief* an agent
    forms from what it has seen, not a fact the market publishes. The part
    anchored to the starting price is the agent's prior sense of value.
    """
    fallback = Decimal(engine.config.economy.initial_reference_price.get(good.value, Decimal(10)))
    trades = engine.market.last_trades(good)
    if not trades:
        return fallback.quantize(PRICE_QUANTUM)
    recent = sum((t.price for t in trades), Decimal(0)) / len(trades)
    belief = (1 - REFERENCE_ANCHOR) * recent + REFERENCE_ANCHOR * fallback
    return belief.quantize(PRICE_QUANTUM)


def starting_price(engine: Engine, good: Good) -> Decimal:
    """The good's starting price: the agent's prior sense of its value."""
    return Decimal(engine.config.economy.initial_reference_price.get(good.value, Decimal(10)))


def _scaled(price: Decimal, factor: Decimal) -> Decimal:
    return max(PRICE_QUANTUM, (price * factor).quantize(PRICE_QUANTUM))


def ask_price(
    engine: Engine, good: Good, reference: Decimal, pressure: Decimal = Decimal(0)
) -> Decimal:
    """What to charge for ``good``, given the state of the book.

    Selling into a resting bid rather than undercutting a trailing mean is what
    makes price discovery two-sided. If every seller simply shaded the recent
    average, the recorded price would drift downward at the shading rate no
    matter how scarce the good became, and a scarcity shock could never show up
    as a price rise. Selling at the bid instead lets demand set the price when
    demand is what is binding.

    ``pressure`` is how badly the seller needs to clear stock. It erodes the
    margin and then cuts into the price, the mirror of a buyer escalating its
    bid: without a way down as well as up, a seller holding stock nobody wants
    simply stops trading and the market seizes instead of clearing lower.

    A resting bid is taken only when it beats what the seller would otherwise
    ask. Taking any bid at all makes the buyer's price the trade price on every
    crossing, which hands price formation entirely to one side.
    """
    target = _scaled(
        reference,
        Decimal(1) + SELLER_MARGIN - min(pressure, SELLER_MARGIN + MAX_SELLER_DISCOUNT),
    )
    bid = engine.market.best_bid(good)
    if bid is not None and bid >= target:
        return bid
    return target


def bid_price(
    engine: Engine, good: Good, reference: Decimal, escalation: Decimal = Decimal(0)
) -> Decimal:
    """What to offer for ``good``, given the state of the book.

    Taking the resting offer when there is one, and bidding progressively higher
    when there is none, is the other half of two-sided discovery: a buyer that
    cannot source a scarce input must raise its price, which is precisely how a
    supply shock becomes a price signal.
    """
    ask = engine.market.best_ask(good)
    ceiling = min(
        _scaled(reference, Decimal(1) + BUYER_PREMIUM + escalation),
        _scaled(starting_price(engine, good), Decimal(1) + MAX_BID_OVER_START),
    )
    if ask is not None and ask <= ceiling:
        return ask
    return ceiling


class RuleBasedPolicy:
    """Base policy: decides one action for one activation."""

    #: Extra premium per activation spent unable to obtain something needed,
    #: and its cap.
    ESCALATION_STEP = Decimal("0.05")
    #: A seller's matching step is smaller. A buyer that cannot get something
    #: scarce chases it faster than a seller with unsold stock marks it down,
    #: which is what lets a shortage show in the price while a mild surplus
    #: only nudges it: a symmetric pair would drift down at every surplus and
    #: barely move under scarcity.
    CONCESSION_STEP = Decimal("0.02")
    MAX_ESCALATION = Decimal("0.60")

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._blocked: dict[tuple[str, str], int] = {}
        self._offers: dict[tuple[str, str], tuple[int, int | None]] = {}
        self._holdings: dict[tuple[str, str], int] = {}
        #: Potions per trip each adventurer has learned to carry, and what it
        #: carried when it last set out.
        self._needs: dict[tuple[str, str], int] = {}
        self._carried: dict[str, dict[Product, int]] = {}

    # -- want escalation ---------------------------------------------------
    def _escalate(
        self,
        agent_id: str,
        good: Good,
        account: AgentAccount | None = None,
        engine: Engine | None = None,
    ) -> Decimal:
        """Raise this agent's willingness to pay for ``good`` and return it.

        Without escalation on both sides of the book the economy deadlocks
        once input costs rise: a producer refuses to sell below cost while a
        buyer's ceiling is pinned to a stale trailing average, so no price
        exists at which the two will trade and the market simply stops. Letting
        a buyer bid up when it genuinely needs something is what carries a
        resource shock through to finished-good prices.
        """
        if account is not None:
            self._note_holding(account, good, engine)
        key = (agent_id, good.value)
        self._blocked[key] = self._blocked.get(key, 0) + 1
        return min(self.ESCALATION_STEP * (self._blocked[key] - 1), self.MAX_ESCALATION)

    def _note_holding(
        self, account: AgentAccount, good: Good, engine: Engine | None = None
    ) -> None:
        """Reset escalation once the agent actually received ``good``.

        Escalation is for a buyer that cannot get filled. One whose holding went
        up since it last bid was filled, and carrying its raised bid into the
        next purchase would ratchet the price with nothing scarce. A holding
        can be used up before the next bid (potions go on a trip), so a bought
        unit since the last bid counts as filled too.
        """
        key = (account.agent_id, f"held:{good.value}")
        held = account.quantity(good)
        if held > self._holdings.get(key, held):
            self._satisfied(account.agent_id, good)
        self._holdings[key] = held
        if engine is not None:
            bought_key = (account.agent_id, f"bought:{good.value}")
            bought = sum(
                t.quantity
                for t in engine.market.trades_of(account.agent_id, len(engine.market.trades))
                if t.good == good and t.buyer_id == account.agent_id
            )
            if bought > self._holdings.get(bought_key, bought):
                self._satisfied(account.agent_id, good)
            self._holdings[bought_key] = bought

    def _satisfied(self, agent_id: str, good: Good) -> None:
        self._blocked.pop((agent_id, good.value), None)

    def _unsold(self, account: AgentAccount, good: Good) -> Decimal:
        """How far to shade the next offer of ``good``: down if unsold, up if selling.

        The seller's counterpart of buyer escalation. Without it a market with
        stock and bids a little apart simply stops: sellers anchor on the last
        trade, buyers do not reach it, nothing trades, and the last trade -- the
        anchor -- never updates. Each renewed offer that finds the holding
        undiminished concedes one step. Each that finds it sold since asks one
        step more (a negative discount): a seller whose goods go as soon as
        they are offered is underpricing them, and a shortage only reaches the
        price if sellers notice it.
        """
        key = (account.agent_id, f"sell:{good.value}")
        held = account.quantity(good)
        count, last = self._offers.get(key, (0, None))
        if last is not None and held >= last:
            count = max(count, 0) + 1
        elif last is not None:
            count = min(count, 0) - 1
        self._offers[key] = (count, held)
        return max(-MAX_SELLER_PREMIUM, min(self.CONCESSION_STEP * count, MAX_SELLER_DISCOUNT))

    def decide(self, engine: Engine, account: AgentAccount, rng: random.Random) -> AgentAction:
        raise NotImplementedError

    def _limit(self) -> int:
        return self.config.market.max_orders_per_activation

    @staticmethod
    def _bundle(orders: list[AgentAction], reason: str) -> AgentAction | None:
        """One activation's market orders: the order itself if one, else place_orders."""
        if not orders:
            return None
        if len(orders) == 1:
            return orders[0]
        return AgentAction(
            action=ActionType.PLACE_ORDERS,
            orders=tuple(orders),
            priority=max(
                (o.priority for o in orders),
                key=lambda p: (Priority.LOW, Priority.MEDIUM, Priority.HIGH).index(p),
            ),
            short_reason=reason,
        )


class AdventurerPolicy(RuleBasedPolicy):
    """Gathers raw materials, maintains equipment, sells surplus (design s4.1)."""

    def decide(self, engine: Engine, account: AgentAccount, rng: random.Random) -> AgentAction:
        """Trade while resting after a trip, then set out again.

        Back in town the adventurer rests (``rest_ticks``) and trades: equips,
        replaces gear, restocks potions, sells its haul. Once rested it sets
        out, whatever orders are still open: they stay on the book while it is
        away. If it got no chance to trade while resting it takes one trade
        action first, and no more, so it neither leaves unequipped for want of
        a turn nor gets stuck in town trading.
        """
        if account.expedition is not None:
            return wait("on expedition")
        self._learn_needs(account)

        agent_id = account.agent_id
        subclass = AdventurerSubclass(account.specialization)
        resting = account.is_resting(engine.tick)
        town = (agent_id, "town_actions")
        traded = self._blocked.get(town, 0)

        if not resting and traded >= 1:
            return self._depart(engine, account, subclass, rng)

        action = self._trade(engine, account, subclass, engine.tick)
        if action is not None:
            self._blocked[town] = traded + 1
            return action
        if resting:
            return wait("resting after an expedition")
        return self._depart(engine, account, subclass, rng)

    def _depart(
        self,
        engine: Engine,
        account: AgentAccount,
        subclass: AdventurerSubclass,
        rng: random.Random,
    ) -> AgentAction:
        self._blocked[(account.agent_id, "town_actions")] = 0
        # What it carries now, to see after the trip how much it used.
        self._carried[account.agent_id] = {
            p: account.quantity(p) for p in self._potions(subclass) if account.quantity(p) > 0
        }
        # Zone choice mixes the subclass's affinity with its own draw, so
        # adventurers of one subclass do not move in lockstep.
        return AgentAction(
            action=ActionType.START_EXPEDITION,
            target=self._choose_zone(engine, subclass, rng),
            short_reason="gather raw materials",
        )

    def _trade(
        self, engine: Engine, account: AgentAccount, subclass: AdventurerSubclass, tick: int
    ) -> AgentAction | None:
        """This return's trading, in one action: None if there is nothing to do.

        Equipping is an action of its own. Everything else is a market order,
        and they all go together: replacing gear, restocking potions, and
        putting the whole haul up for sale.
        """
        # Durable gear, most valuable first: equipped before the trip, re-bought
        # when it breaks.
        gear = self.config.expeditions.gear.get(subclass, {})
        durable = sorted(
            (p for p in gear if p in self.config.economy.durability),
            key=lambda p: (-gear[p], p.value),
        )
        mine = engine.market.open_orders_of(account.agent_id)
        # One resting bid per good: a new one every return locks the
        # adventurer's gold in orders and it ends up setting out short.
        pending = {order.good for order in mine if order.side is OrderSide.BUY}
        offered = {order.good for order in mine if order.side is OrderSide.SELL}

        # Equip gear already owned before buying more.
        for item in durable:
            if not account.is_equipped_with(item) and account.available_quantity(item) > 0:
                self._satisfied(account.agent_id, item)
                return AgentAction(
                    action=ActionType.EQUIP,
                    target=item.value,
                    priority=Priority.HIGH,
                    short_reason="equip owned gear before departing",
                )

        orders: list[AgentAction] = []
        limit = self._limit()
        budget = account.available_gold

        # Needs first. Items are bought most valuable first, each as soon as it
        # is affordable.
        for rank, item in enumerate(durable):
            if len(orders) >= limit or account.is_equipped_with(item) or item in pending:
                continue
            # Gear nobody is selling is not worth every return: after a few
            # lapsed bids the adventurer stops asking for a while.
            skip = (account.agent_id, f"skip:{item.value}")
            if tick < self._blocked.get(skip, -1):
                continue
            if self._blocked.get((account.agent_id, item.value), 0) >= GEAR_PATIENCE:
                self._blocked[skip] = tick + GEAR_RETRY_TICKS
                self._blocked[(account.agent_id, item.value)] = 0
                continue
            price = bid_price(
                engine,
                item,
                reference_price(engine, item),
                self._escalate(account.agent_id, item, account, engine),
            )
            if budget >= price:
                budget -= price
                orders.append(
                    AgentAction(
                        action=ActionType.BUY,
                        target=item.value,
                        quantity=1,
                        price=price,
                        priority=Priority.HIGH if rank == 0 else Priority.MEDIUM,
                        short_reason="replace missing gear",
                    )
                )

        # Restock potions. Without this nothing ever buys a potion, the
        # alchemist has no market, and every trip after the starting stock
        # sets out short. Every potion that is short and not already bid for:
        # a pending bid for one must not keep the other from being restocked.
        for consumable in self._consumables_to_restock(engine, account, pending):
            if len(orders) >= limit:
                break
            price = bid_price(
                engine,
                consumable,
                reference_price(engine, consumable),
                self._escalate(account.agent_id, consumable, account, engine),
            )
            # Enough for the next trip, a demand surge included, and never
            # fewer than the usual lot.
            short = self._trip_need(account, consumable) - account.available_quantity(consumable)
            wanted = max(CONSUMABLE_RESTOCK_QUANTITY, short)
            quantity = min(wanted, int(budget / price)) if price > 0 else 0
            if quantity > 0:
                budget -= price * quantity
                orders.append(
                    AgentAction(
                        action=ActionType.BUY,
                        target=consumable.value,
                        quantity=quantity,
                        price=price,
                        priority=Priority.HIGH,
                        short_reason="restock consumables",
                    )
                )

        # Then the haul, all of it. An adventurer has no use for raw materials.
        # Into a waiting bid when there is one at or above the recent price:
        # taking lowball bids would hand scarce stock to whoever bids least, and
        # the price could never rise with scarcity. Otherwise as an offer that
        # concedes step by step while unsold.
        haul = sorted(
            (r for r in Resource if account.available_quantity(r) > 0 and r not in offered),
            key=lambda r: (-account.available_quantity(r), r.value),
        )
        for resource in haul:
            if len(orders) >= limit:
                break
            held = account.available_quantity(resource)
            floor = self._hold_floor(engine, account, resource)
            bid = engine.market.best_bid(resource)
            if bid is not None and bid >= max(reference_price(engine, resource), floor):
                price, reason = bid, "sell to a waiting buyer"
            else:
                excess = held - ADVENTURER_STOCK_THRESHOLD
                pressure = (
                    OVERSTOCK_STEP * Decimal(max(0, excess)) / max(1, ADVENTURER_STOCK_THRESHOLD)
                )
                pressure += self._unsold(account, resource)
                price = ask_price(engine, resource, reference_price(engine, resource), pressure)
                price, reason = max(price, floor), "convert surplus haul to gold"
            orders.append(
                AgentAction(
                    action=ActionType.SELL,
                    target=resource.value,
                    quantity=min(held, MAX_LOT),
                    price=price,
                    short_reason=reason,
                )
            )

        return self._bundle(orders, "restock and sell the haul")

    def _hold_floor(self, engine: Engine, account: AgentAccount, resource: Resource) -> Decimal:
        """The lowest price worth selling at now: HOLD_FRACTION of the starting
        price while the price has been that low for under HOLD_PATIENCE ticks,
        otherwise nothing."""
        start = Decimal(self.config.economy.initial_reference_price.get(resource.value, 0))
        threshold = (start * HOLD_FRACTION).quantize(PRICE_QUANTUM)
        key = (account.agent_id, f"hold:{resource.value}")
        if reference_price(engine, resource) >= threshold:
            self._blocked.pop(key, None)
            return Decimal(0)
        since = self._blocked.setdefault(key, engine.tick)
        return threshold if engine.tick - since < HOLD_PATIENCE else Decimal(0)

    def _potions(self, subclass: AdventurerSubclass) -> list[Product]:
        """The consumables in this subclass's gear: used up on each trip."""
        gear = self.config.expeditions.gear.get(subclass, {})
        return sorted(
            (p for p in gear if p not in self.config.economy.durability),
            key=lambda p: p.value,
        )

    def _trip_need(self, account: AgentAccount, product: Product) -> int:
        """Units of a potion the agent expects a trip to use up.

        Learned from its own trips, never read from the shock: what the last
        trip consumed, or one more when it used up all it carried (it may have
        needed more). One until it has seen a trip.
        """
        return self._needs.get((account.agent_id, product.value), CONSUMABLE_RESTOCK_THRESHOLD)

    def _learn_needs(self, account: AgentAccount) -> None:
        """After a trip, compare what it carried with what is left."""
        carried = self._carried.pop(account.agent_id, None)
        if carried is None:
            return
        for product, before in carried.items():
            left = account.quantity(product)
            used = before - left
            if used <= 0:
                continue
            need = used + 1 if left == 0 else used
            self._needs[(account.agent_id, product.value)] = max(
                CONSUMABLE_RESTOCK_THRESHOLD, min(need, MAX_POTIONS_PER_TRIP)
            )

    def _consumables_to_restock(
        self, engine: Engine, account: AgentAccount, pending: set[Good]
    ) -> list[Product]:
        """The potions of this adventurer's gear it holds too few of for a trip.

        In the order of ``_potions`` (by name), skipping any it already has a
        resting bid for.
        """
        subclass = AdventurerSubclass(account.specialization)
        return [
            product
            for product in self._potions(subclass)
            if product not in pending
            and account.available_quantity(product) < self._trip_need(account, product)
        ]

    def _choose_zone(self, engine: Engine, subclass: AdventurerSubclass, rng: random.Random) -> str:
        """Prefer zones rich in what this subclass gathers well and what is selling.

        Each resource counts by its recent price against its starting one. With
        affinity alone, supply never answers the market: adventurers keep
        hauling a good whose price has collapsed under a glut, and the glut
        only grows. Weighting by price is the quantity side of supply and
        demand: a cheap good is gathered less, a dear one more.
        """
        settings = self.config.expeditions.subclasses.get(subclass)
        affinity = settings.resource_affinity if settings else {}
        starting = self.config.economy.initial_reference_price

        weights: list[tuple[str, int]] = []
        for zone in engine.zones.zones():
            score = Decimal(1)
            for resource, rate in zone.spawn_per_tick.items():
                if zone.stock(resource) > 0:
                    base = Decimal(starting.get(resource.value, 0)) or Decimal(1)
                    relative = reference_price(engine, resource) / base
                    score += rate * affinity.get(resource, Decimal(1)) * relative
            weights.append((zone.zone_id, int(score * 100)))

        total = sum(weight for _, weight in weights)
        if total <= 0:
            return engine.zones.zone_ids[0]
        roll = rng.randrange(total)
        cumulative = 0
        for zone_id, weight in weights:
            cumulative += weight
            if roll < cumulative:
                return zone_id
        return weights[-1][0]  # pragma: no cover -- unreachable while total > 0


class ArtisanPolicy(RuleBasedPolicy):
    """Crafts within its specialization, sources inputs, sells output (s4.2).

    Two behaviours matter for the experiment's economics. It **escalates its
    bid** the longer it is blocked on a missing input, which is what turns a
    supply shock into a rising price rather than a silent stall. And it prices
    output at cost plus a markup, so a rise in input prices propagates to
    the finished good -- the resource-to-product propagation design v1.0 s11.3
    pre-registers as a secondary analysis.
    """

    #: Markup over input cost. Selling below cost is how an artisan goes broke.
    COST_MARKUP = Decimal("1.2")
    #: Stop making a product once this many are already sitting unsold.
    MAX_FINISHED_STOCK = 3
    #: The same for consumables (potions), which every trip uses up.
    MAX_CONSUMABLE_STOCK = 4

    def _batch_size(self, engine: Engine, account: AgentAccount, product: Product) -> int:
        """How many units to craft in this one activation.

        Crafting a single unit per activation caps an artisan's throughput at
        one unit per action, and with one action per macro-cycle the whole
        economy's absorption falls an order of magnitude below what adventurers
        gather. The result is chronic oversupply, a downward price drift, and a
        market too thin to price a shock. The engine already validates the batch
        against capacity and stock, so the agent may ask for as much as its
        capacity and inputs allow.

        The batch counts recipe runs, each yielding ``output_quantity`` units,
        so the room left under the stock cap is converted from units to runs.
        """
        recipe = self.config.economy.recipes[product]
        from_inputs = min(
            account.available_quantity(resource) // needed
            for resource, needed in recipe.inputs.items()
        )
        room = self._stock_cap(product) - account.quantity(product)
        runs_for_room = -(-room // recipe.output_quantity)
        return max(1, min(engine.production.capacity_of(account), from_inputs, runs_for_room))

    def _stock_cap(self, product: Product) -> int:
        """Finished stock worth holding: one batch, or MAX_FINISHED_STOCK if less.

        Production follows sales. An artisan that keeps several units of each
        line in stock makes more than wears out, remarks the surplus ever
        cheaper, and drags down the price of its own inputs with it: a
        deflation that starts in finished goods and runs back down the chain.
        """
        output = self.config.economy.recipes[product].output_quantity
        if product not in self.config.economy.durability:
            # Potions are used up on every trip by every adventurer: the line
            # keeps enough on hand for several buyers, not one.
            cap = max(output, self.MAX_CONSUMABLE_STOCK)
        else:
            cap = min(self.MAX_FINISHED_STOCK, output)
        # Never past the engine's craft limit: held at it, the policy kept
        # choosing to craft, was refused every time, and never reached the
        # selling branch, so it stopped selling and buying altogether.
        limit = self.config.economy.craft_stock_limit
        return cap if limit is None else min(cap, limit)

    def _input_cost(self, engine: Engine, product: Product) -> Decimal:
        recipe = self.config.economy.recipes[product]
        return sum(
            (
                reference_price(engine, resource) * quantity
                for resource, quantity in recipe.inputs.items()
            ),
            Decimal(0),
        )

    def _unit_cost(self, engine: Engine, product: Product) -> Decimal:
        """What the inputs of one unit of ``product`` cost: a batch over its output."""
        recipe = self.config.economy.recipes[product]
        return self._input_cost(engine, product) / recipe.output_quantity

    def _input_ceiling(self, engine: Engine, product: Product, resource: Resource) -> Decimal:
        """The most one unit of ``resource`` is worth as an input to ``product``.

        What the product sells for, less the other inputs at their recent
        prices, spread over the units of this input the recipe needs.
        """
        recipe = self.config.economy.recipes[product]
        value = reference_price(engine, product) * recipe.output_quantity
        others = sum(
            (reference_price(engine, r) * q for r, q in recipe.inputs.items() if r is not resource),
            Decimal(0),
        )
        return ((value - others) / recipe.inputs[resource]).quantize(PRICE_QUANTUM)

    def _sourcing_priority(self, engine: Engine, product: Product) -> tuple[int, Decimal, str]:
        """Sort key: products with a resting bid first, then by expected margin."""
        bid = engine.market.best_bid(product)
        value = bid if bid is not None else reference_price(engine, product)
        margin = value - self._unit_cost(engine, product)
        return (0 if bid is not None else 1, -margin, product.value)

    def decide(self, engine: Engine, account: AgentAccount, rng: random.Random) -> AgentAction:
        specialization = engine.production.specialization_of(account)
        if specialization is None:
            return wait("no specialization")

        products = engine.production.products_for(specialization)

        # Craft what is covered by stock and not already piling up unsold.
        #
        # Producing whatever comes first in the recipe table, regardless of what
        # is selling, is how an artisan ends a run holding a dozen unsold
        # weapons and no cash: it converts saleable inputs into stock nobody
        # wants, and starves itself of the gold needed to buy more inputs.
        # Design v1.0 s11.1 measures exactly this as overproduction.
        if engine.production.capacity_of(account) > 0:
            buildable = [
                product
                for product in products
                if not engine.production.missing_inputs(account, product)
                # Stock already offered counts: it is unsold all the same.
                and account.quantity(product) < self._stock_cap(product)
            ]
            if buildable:
                product = min(buildable, key=lambda p: (account.available_quantity(p), p.value))
                for resource in self.config.economy.recipes[product].inputs:
                    self._satisfied(account.agent_id, resource)
                return AgentAction(
                    action=ActionType.PRODUCE,
                    target=product.value,
                    quantity=self._batch_size(engine, account, product),
                    short_reason="build the line that is shortest of stock",
                )

        # Otherwise trade, all at once: offer every finished good and bid for
        # every input the next batches lack.
        orders: list[AgentAction] = []
        limit = self._limit()

        # Sell finished goods at cost plus a markup, never below cost.
        for finished in sorted(products, key=lambda p: (-account.available_quantity(p), p.value)):
            if len(orders) >= limit or account.available_quantity(finished) <= 0:
                continue
            reference = reference_price(engine, finished)
            # The markup gives way while the stock sits unsold, down to cost and
            # never below: an artisan holding goods nobody takes at its price has
            # its gold locked in them and cannot buy the next inputs.
            concession = self._unsold(account, finished)
            markup = max(Decimal(1), self.COST_MARKUP - concession)
            floor = (self._unit_cost(engine, finished) * markup).quantize(PRICE_QUANTUM)
            price = max(ask_price(engine, finished, reference, concession), floor)
            orders.append(
                AgentAction(
                    action=ActionType.SELL,
                    target=finished.value,
                    quantity=min(account.available_quantity(finished), MAX_LOT),
                    price=price,
                    short_reason="move finished stock at cost plus markup",
                )
            )

        # Source the inputs that block production, bidding higher the longer
        # they stay blocked. Lines with buyers waiting and the widest margin
        # come first: sourcing whatever line comes first in the recipe table
        # leaves an alchemist chasing scarce wood for a staff while buyers bid
        # twice the cost of a potion made from cheap herbs.
        pending = {
            order.good
            for order in engine.market.open_orders_of(account.agent_id)
            if order.side is OrderSide.BUY
        }
        budget = account.available_gold
        for product in sorted(products, key=lambda p: self._sourcing_priority(engine, p)):
            # No inputs for a line whose output is not selling: that gold would
            # only turn into more stock nobody takes.
            if account.quantity(product) >= self._stock_cap(product):
                continue
            for resource, units in engine.production.missing_inputs(account, product).items():
                # One resting bid per input. Stacking a new one every activation
                # locks all the agent's gold in orders and leaves it idle.
                if len(orders) >= limit or resource in pending:
                    continue
                price = bid_price(
                    engine,
                    resource,
                    reference_price(engine, resource),
                    self._escalate(account.agent_id, resource, account, engine),
                )
                # Never pay more for an input than the product it goes into can
                # return: past that, buying it loses money on every unit made.
                ceiling = self._input_ceiling(engine, product, resource)
                if ceiling < PRICE_QUANTUM:
                    continue
                price = min(price, ceiling)
                quantity = min(units + 2, int(budget / price)) if price > 0 else 0
                if quantity <= 0:
                    continue
                budget -= price * quantity
                pending.add(resource)
                orders.append(
                    AgentAction(
                        action=ActionType.BUY,
                        target=resource.value,
                        quantity=quantity,
                        price=price,
                        priority=Priority.HIGH,
                        short_reason="input shortage blocks production",
                    )
                )

        return self._bundle(orders, "sell finished stock and source inputs") or wait(
            "no inputs affordable and nothing to sell"
        )


def policy_for(role: Role, config: SimulationConfig) -> RuleBasedPolicy:
    """The policy governing a role."""
    match role:
        case Role.ADVENTURER:
            return AdventurerPolicy(config)
        case Role.ARTISAN:
            return ArtisanPolicy(config)
    raise ValueError(f"no rule-based policy for role {role}")  # pragma: no cover
