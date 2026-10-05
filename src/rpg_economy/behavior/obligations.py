"""What each role must do, as checks over one observation and one answer.

An obligation says when it applies (from the observation alone: what the agent
holds, what the book shows, whether it can set out) and whether an answer
meets it. The same list scores the behaviour check's answers and audits the
activations of a real run, so a run is judged by the rules the check tests.

The rules are the economy's, stated once: an adventurer sells the raw materials
buyers wait for and gets the gear it needs to set out; an artisan crafts and
sells what buyers want, buys the inputs it lacks for that, raises a bid that
failed, and neither sells below cost nor crafts what does not sell; and every
agent only offers what it holds and bids what it can pay. Where two rules would
ask for different actions in one activation, the earlier in each role's list
takes precedence and the later does not apply, so an answer is never asked for
two things at once.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Product, Resource

RAW = frozenset(r.value for r in Resource)
PRODUCTS = frozenset(p.value for p in Product)
#: Below this share of the input cost a sale counts as below cost: an artisan
#: may shade its price a little to sell, not undercut its inputs.
COST_TOLERANCE = Decimal("0.9")
#: A sale priced above this multiple of the best bid is not a sale to that bid.
NEAR_BID = Decimal("1.5")
#: Units of a product held with no bid from which crafting more is flagged.
UNSOLD_FLAG = 3
_BUYING = re.compile(r"^Buying: (\w+): your last \d+ bid\(s\) at ([0-9., ]+) expired")


@dataclass(frozen=True)
class GearOffer:
    """One item an adventurer lacks, as its headline states it."""

    item: str
    #: Held but not equipped: equipping it is all it takes.
    held: bool
    ask: Decimal | None
    #: What it costs a trip: price over durability, or a potion's price.
    cost_per_trip: Decimal | None
    #: It adds more to a trip than it costs, in every zone (or it is held).
    pays: bool


def _dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _value(item: Any) -> str:
    return str(getattr(item, "value", item))


class View:
    """One observation (the logged payload form), with the facts the rules ask."""

    def __init__(
        self,
        config: SimulationConfig,
        observation: dict[str, Any],
        task: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.obs = observation
        self.task = task
        self.role = observation.get("role")
        self.spec = observation.get("specialization")
        self.tick = int(observation.get("tick", 0))
        self.gold = _dec(observation.get("available_gold")) or Decimal(0)
        self.market = {m["good"]: m for m in observation.get("market", [])}
        self.open_orders = list(observation.get("open_orders") or [])
        self.status = list(observation.get("status") or [])

    # -- holdings and state --------------------------------------------------
    def held(self, good: str) -> int:
        """Units the agent can still sell, equip or use (not on its own orders)."""
        return int((self.obs.get("available_inventory") or {}).get(good, 0))

    def owned(self, good: str) -> int:
        return int((self.obs.get("inventory") or {}).get(good, 0))

    @property
    def is_adventurer(self) -> bool:
        return self.role == "adventurer"

    @property
    def is_artisan(self) -> bool:
        return self.role == "artisan"

    @property
    def away(self) -> bool:
        return bool(self.obs.get("is_away"))

    @property
    def resting(self) -> bool:
        return self.obs.get("can_set_out_from_tick") is not None

    @property
    def can_set_out(self) -> bool:
        return self.is_adventurer and not self.away and not self.resting

    def gear_offers(self) -> list[GearOffer]:
        """The gear the adventurer lacks, as the observation states it."""
        return [
            GearOffer(
                item=fact["item"],
                held=bool(fact.get("held")),
                ask=_dec(fact["ask"]),
                cost_per_trip=_dec(fact["cost_per_trip"]),
                pays=bool(fact["pays"]),
            )
            for fact in self.obs.get("gear") or []
        ]

    @property
    def gear(self) -> set[str]:
        return set((self.obs.get("expedition_gear") or {}).keys())

    # -- market --------------------------------------------------------------
    def bid(self, good: str) -> Decimal | None:
        return _dec(self.market.get(good, {}).get("best_bid"))

    def ask(self, good: str) -> Decimal | None:
        return _dec(self.market.get(good, {}).get("best_ask"))

    def last_price(self, good: str) -> Decimal:
        recent = self.market.get(good, {}).get("recent_prices") or []
        if recent:
            return Decimal(str(recent[-1]))
        return Decimal(str(self.config.economy.initial_reference_price[good]))

    def has_open(self, side: str, good: str, *, at_least: Decimal | None = None) -> bool:
        return any(
            o["side"] == side
            and o["good"] == good
            and (at_least is None or Decimal(str(o["limit_price"])) >= at_least)
            for o in self.open_orders
        )

    # -- production ----------------------------------------------------------
    def recipes(self) -> dict[str, dict[str, int]]:
        """This artisan's products -> their inputs per run."""
        if not self.is_artisan:
            return {}
        return {
            _value(product): {_value(r): q for r, q in recipe.inputs.items()}
            for product, recipe in self.config.economy.recipes.items()
            if _value(recipe.specialization) == self.spec
        }

    def unit_cost(self, product: str) -> Decimal:
        recipe = self.config.economy.recipes[Product(product)]
        total = sum((self.last_price(_value(r)) * q for r, q in recipe.inputs.items()), Decimal(0))
        return (total / recipe.output_quantity).quantize(Decimal("0.01"))

    def runs(self, product: str) -> int:
        inputs = self.recipes()[product]
        return min(self.held(good) // q for good, q in inputs.items())

    def wanted(self, product: str) -> bool:
        """A buyer waits for it above its input cost and nobody offers it."""
        bid = self.bid(product)
        return bid is not None and self.ask(product) is None and bid >= self.unit_cost(product)

    @property
    def capacity(self) -> int:
        return int(self.obs.get("production_capacity") or 0)


# -- reading an answer ---------------------------------------------------------
def orders(action: dict[str, Any]) -> list[dict[str, Any]]:
    """The market orders an answer places: a place_orders' list, or the action itself."""
    if action.get("action") == "place_orders":
        return list(action.get("orders") or [])
    return [action]


def sells(action: dict[str, Any], goods: set[str] | frozenset[str] | None = None):
    return [
        o
        for o in orders(action)
        if o.get("action") == "sell" and (goods is None or o.get("target") in goods)
    ]


def buys(action: dict[str, Any], goods: set[str] | frozenset[str] | None = None):
    return [
        o
        for o in orders(action)
        if o.get("action") == "buy" and (goods is None or o.get("target") in goods)
    ]


def price(order: dict[str, Any]) -> Decimal:
    return _dec(order.get("price")) or Decimal(0)


def cancels(action: dict[str, Any]) -> bool:
    return any(o.get("action") == "cancel_order" for o in orders(action))


@dataclass(frozen=True)
class Obligation:
    name: str
    role: str  # adventurer | artisan | any
    #: Share of answers that must meet it, where it applies, for a check to pass.
    threshold: float
    applies: Callable[[View], bool]
    #: True or False; None when the answer does not do what the rule is about
    #: (a pricing rule, for an answer that sells nothing): it is not counted.
    met: Callable[[View, dict[str, Any]], bool | None]
    summary: str
    #: Judged on the share over all the answers it counted, not per situation:
    #: a rule about how often something happens, such as asking above the bid.
    aggregate: bool = False

    def for_role(self, role: str | None) -> bool:
        return self.role == "any" or self.role == role


# -- adventurer rules -----------------------------------------------------------
def _held_gear(v: View) -> set[str]:
    # Only logs before prompts 2.36 list held gear: setting out now equips it.
    return {o.item for o in v.gear_offers() if o.held} if not v.away else set()


def _gear_to_buy(v: View) -> set[str]:
    """Offered items that pay for themselves in every zone and that it can afford."""
    return {
        o.item
        for o in v.gear_offers()
        if not o.held
        and o.ask is not None
        and o.pays
        and v.gold >= o.ask
        and not v.has_open("buy", o.item, at_least=o.ask)
    }


def _gear_to_bid_for(v: View) -> set[str]:
    """Items nobody offers that would pay at their last price, with no bid of its own."""
    return {
        o.item
        for o in v.gear_offers()
        if not o.held
        and o.ask is None
        and o.pays
        and not v.has_open("buy", o.item)
        and v.gold >= (o.cost_per_trip or Decimal(0))
    }


def _kept_gear(v: View) -> set[str]:
    """Gear and potions it holds unreserved: what its own trips use."""
    return {g for g in v.gear if v.held(g) > 0}


def _wanted_raw(v: View) -> set[str]:
    return {g for g in RAW if v.held(g) > 0 and v.bid(g) is not None}


def _sells_wanted_raw_applies(v: View) -> bool:
    return (
        # Selling and buying go together in one place_orders; equipping does not.
        v.is_adventurer and not v.can_set_out and bool(_wanted_raw(v)) and not _held_gear(v)
    )


def _sells_wanted_raw_met(v: View, a: dict[str, Any]) -> bool:
    return bool(sells(a, _wanted_raw(v)))


def _ready_met(v: View, a: dict[str, Any]) -> bool:
    return (
        a.get("action") == "start_expedition"
        or (a.get("action") == "equip" and a.get("target") in _held_gear(v))
        or bool(sells(a, _wanted_raw(v)))
        or bool(buys(a, v.gear))
    )


def _scarce_raw(v: View) -> set[str]:
    """Raw materials it holds that buyers bid for and nobody offers."""
    return {g for g in _wanted_raw(v) if v.ask(g) is None}


def _asks_above_bid_met(v: View, a: dict[str, Any]) -> bool | None:
    offers = sells(a, _scarce_raw(v))
    if not offers:
        return None
    return any(v.bid(o["target"]) < price(o) <= v.bid(o["target"]) * NEAR_BID for o in offers)


# -- artisan rules -----------------------------------------------------------------
def _wanted_stock(v: View) -> set[str]:
    return {p for p in v.recipes() if v.held(p) > 0 and v.wanted(p)}


def _wanted_craftable(v: View) -> set[str]:
    return {p for p in v.recipes() if v.held(p) == 0 and v.wanted(p) and v.runs(p) >= 1}


def _missing_inputs(v: View) -> set[str]:
    """Inputs of a wanted product it cannot craft, with no bid of its own for them."""
    missing: set[str] = set()
    for product, inputs in v.recipes().items():
        if v.held(product) or not v.wanted(product) or v.runs(product) >= 1:
            continue
        short = {g for g, q in inputs.items() if v.held(g) < q}
        if any(v.has_open("buy", g) for g in short):
            continue
        cost = sum((v.last_price(g) * (inputs[g] - v.held(g)) for g in short), Decimal(0))
        if v.gold >= cost:
            missing |= short
    return missing


def _sells_wanted_stock_applies(v: View) -> bool:
    return v.is_artisan and bool(_wanted_stock(v))


def _sells_wanted_stock_met(v: View, a: dict[str, Any]) -> bool:
    return any(
        v.unit_cost(o["target"]) * COST_TOLERANCE <= price(o) <= v.bid(o["target"]) * NEAR_BID
        for o in sells(a, _wanted_stock(v))
    )


def _crafts_wanted_applies(v: View) -> bool:
    return (
        v.is_artisan
        and v.capacity > 0
        and bool(_wanted_craftable(v))
        and not _sells_wanted_stock_applies(v)
    )


def _buys_input_applies(v: View) -> bool:
    return (
        v.is_artisan
        and bool(_missing_inputs(v))
        and not _sells_wanted_stock_applies(v)
        and not _crafts_wanted_applies(v)
    )


def _failed_bids(v: View) -> dict[str, Decimal]:
    """Inputs whose recent bids expired unbought -> the highest of those bids.

    Asked only of an artisan with nothing of its own to sell, or for an input
    of a product buyers want: with stock on hand, repricing that stock is as
    fair a use of the turn.
    """
    idle = not any(v.owned(product) for product in v.recipes())
    found: dict[str, Decimal] = {}
    for line in v.status:
        match = _BUYING.match(line)
        if not match:
            continue
        # The highest bid that failed: the line suggests a step above it.
        top = max(Decimal(b) for b in match.group(2).split(", "))
        if v.has_open("buy", match.group(1), at_least=top):
            continue
        good = match.group(1)
        if idle or any(
            good in inputs and v.wanted(product) for product, inputs in v.recipes().items()
        ):
            found[good] = top
    return found


def _raises_bid_met(v: View, a: dict[str, Any]) -> bool:
    return any(price(o) > top for g, top in _failed_bids(v).items() for o in buys(a, {g}))


def _unsold(v: View) -> set[str]:
    return {p for p in v.recipes() if v.owned(p) >= UNSOLD_FLAG and v.bid(p) is None}


def _no_sale_below_cost_met(v: View, a: dict[str, Any]) -> bool:
    own = set(v.recipes())
    return all(price(o) >= v.unit_cost(o["target"]) * COST_TOLERANCE for o in sells(a, own))


def _produces_possible_met(v: View, a: dict[str, Any]) -> bool:
    if a.get("action") != "produce":
        return True
    product = a.get("target")
    if product not in v.recipes():
        return False
    quantity = int(a.get("quantity") or 1)
    return v.runs(product) >= quantity and v.capacity >= quantity


# -- rules for everyone --------------------------------------------------------------
def _sells_within_stock_met(v: View, a: dict[str, Any]) -> bool:
    totals: dict[str, int] = {}
    for o in sells(a):
        totals[o.get("target")] = totals.get(o.get("target"), 0) + int(o.get("quantity") or 0)
    # A cancel in the same answer frees what it held; which one is not logged.
    limit = v.owned if cancels(a) else v.held
    return all(q <= limit(g) for g, q in totals.items())


def _buys_within_gold_met(v: View, a: dict[str, Any]) -> bool:
    spend = sum((price(o) * int(o.get("quantity") or 0) for o in buys(a)), Decimal(0))
    gold = (_dec(v.obs.get("gold")) or Decimal(0)) if cancels(a) else v.gold
    return spend <= gold


def _no_self_cross_met(v: View, a: dict[str, Any]) -> bool:
    if cancels(a):
        return True
    for o in orders(a):
        side, good = o.get("action"), o.get("target")
        for mine in v.open_orders:
            if mine["good"] != good:
                continue
            limit = Decimal(str(mine["limit_price"]))
            if side == "buy" and mine["side"] == "sell" and price(o) >= limit:
                return False
            if side == "sell" and mine["side"] == "buy" and price(o) <= limit:
                return False
    return True


def _equips_only_held_met(v: View, a: dict[str, Any]) -> bool:
    if a.get("action") != "equip":
        return True
    return not v.away and v.held(a.get("target")) > 0


OBLIGATIONS: list[Obligation] = [
    # adventurer
    Obligation(
        "adv_keeps_gear",
        "adventurer",
        0.85,
        # Adventurers sold the potions their next trip would use,
        # undercutting the alchemists; the market now refuses it, and an
        # answer that tries it is a wasted activation.
        lambda v: v.is_adventurer and bool(_kept_gear(v)),
        lambda v, a: not sells(a, _kept_gear(v)),
        "holds gear or potions its trips use: does not sell them",
    ),
    Obligation(
        "adv_buys_paying_gear",
        "adventurer",
        0.75,
        # Asked while resting or away: once it can set out, buying first and
        # setting out compete for one action, and either is fair (adv_ready_acts).
        lambda v: (
            v.is_adventurer and not v.can_set_out and not _held_gear(v) and bool(_gear_to_buy(v))
        ),
        lambda v, a: any(price(o) >= v.ask(o["target"]) for o in buys(a, _gear_to_buy(v))),
        "gear on offer would add more to every trip than it costs a trip, and it can pay: buys it",
    ),
    Obligation(
        "adv_bids_for_paying_gear",
        "adventurer",
        0.4,
        lambda v: (
            v.is_adventurer
            and not v.can_set_out
            and not _held_gear(v)
            and not _gear_to_buy(v)
            and bool(_gear_to_bid_for(v))
        ),
        lambda v, a: bool(buys(a, _gear_to_bid_for(v))),
        "gear nobody offers would pay for itself: bids for it",
    ),
    Obligation(
        "adv_no_set_out_when_unable",
        "adventurer",
        0.85,
        lambda v: v.is_adventurer and not v.away and v.resting,
        lambda v, a: a.get("action") != "start_expedition",
        "resting: does not set out",
    ),
    Obligation(
        "adv_sells_wanted_raw",
        "adventurer",
        0.75,
        _sells_wanted_raw_applies,
        _sells_wanted_raw_met,
        "resting or away, holds raw materials buyers bid for: offers them",
    ),
    Obligation(
        "adv_asks_above_bid_when_scarce",
        "adventurer",
        0.25,
        lambda v: v.is_adventurer and bool(_scarce_raw(v)),
        _asks_above_bid_met,
        "offering a raw material nobody else offers, sometimes asks above the bid",
        aggregate=True,
    ),
    Obligation(
        "adv_ready_acts",
        "adventurer",
        0.75,
        lambda v: v.can_set_out,
        _ready_met,
        "can set out: sets out, equips or buys its gear, or sells wanted raw materials",
    ),
    Obligation(
        "adv_no_raw_buy",
        "adventurer",
        0.85,
        lambda v: v.is_adventurer,
        lambda v, a: not buys(a, RAW),
        "never bids for raw materials (adventurers only sell them)",
    ),
    Obligation(
        "adv_buys_only_its_gear",
        "adventurer",
        0.85,
        lambda v: v.is_adventurer,
        lambda v, a: all(o.get("target") in v.gear for o in buys(a, PRODUCTS)),
        "buys only products its subclass can carry",
    ),
    Obligation(
        "adv_equips_only_held",
        "adventurer",
        0.85,
        lambda v: v.is_adventurer,
        _equips_only_held_met,
        "equips only an item it holds, and not while away",
    ),
    # artisan
    Obligation(
        "art_sells_wanted_stock",
        "artisan",
        0.85,
        _sells_wanted_stock_applies,
        _sells_wanted_stock_met,
        "holds a product buyers bid for above its cost: offers it near that bid",
    ),
    Obligation(
        "art_crafts_wanted",
        "artisan",
        0.85,
        _crafts_wanted_applies,
        lambda v, a: a.get("action") == "produce" and a.get("target") in _wanted_craftable(v),
        "can craft a product buyers wait for: crafts it",
    ),
    Obligation(
        "art_buys_input_for_wanted",
        "artisan",
        0.75,
        _buys_input_applies,
        lambda v, a: bool(buys(a, _missing_inputs(v))),
        "lacks the inputs for a product buyers wait for: bids for them",
    ),
    Obligation(
        "art_raises_failed_bid",
        "artisan",
        0.75,
        lambda v: (
            v.is_artisan
            and bool(_failed_bids(v))
            and not _sells_wanted_stock_applies(v)
            and not _crafts_wanted_applies(v)
        ),
        _raises_bid_met,
        "its bids for an input expired unbought: bids higher",
    ),
    Obligation(
        "art_stops_unsold",
        "artisan",
        0.85,
        lambda v: v.is_artisan and bool(_unsold(v)),
        lambda v, a: not (a.get("action") == "produce" and a.get("target") in _unsold(v)),
        "holds unsold stock of a product with no bid: does not craft more of it",
    ),
    Obligation(
        "art_no_sale_below_cost",
        "artisan",
        0.85,
        lambda v: v.is_artisan and any(v.owned(p) for p in v.recipes()),
        _no_sale_below_cost_met,
        "does not offer its products below their input cost",
    ),
    Obligation(
        "art_produces_only_possible",
        "artisan",
        0.85,
        lambda v: v.is_artisan,
        _produces_possible_met,
        "crafts only what its inputs and capacity allow",
    ),
    # everyone
    Obligation(
        "sells_within_stock",
        "any",
        0.85,
        lambda v: True,
        _sells_within_stock_met,
        "offers only what it holds and has not already offered",
    ),
    Obligation(
        "buys_within_gold",
        "any",
        0.85,
        lambda v: True,
        _buys_within_gold_met,
        "bids only what its free gold can pay",
    ),
    Obligation(
        "no_self_cross",
        "any",
        0.85,
        lambda v: True,
        _no_self_cross_met,
        "does not place an order that would trade with its own resting order",
    ),
    Obligation(
        "task_answered_with_action",
        "any",
        0.75,
        lambda v: v.task is not None,
        lambda v, a: a.get("action") != "wait",
        "given a TASK (A): acts on it rather than waiting",
    ),
]

BY_NAME = {o.name: o for o in OBLIGATIONS}


def applicable(view: View) -> list[Obligation]:
    """The obligations that apply in this observation."""
    return [o for o in OBLIGATIONS if o.for_role(view.role) and o.applies(view)]


def judge(view: View, action: dict[str, Any] | None) -> dict[str, bool | None]:
    """Each applicable obligation -> whether the answer met it.

    An invalid answer (``action`` None) misses every rule, except those whose
    answers are counted only when they do what the rule is about.
    """
    return {
        o.name: o.met(view, action) if action is not None else (None if o.aggregate else False)
        for o in applicable(view)
    }
