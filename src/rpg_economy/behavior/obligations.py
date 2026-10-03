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


def _kept_gear(v: View) -> set[str]:
    """Gear and potions it holds unreserved: what its own trips use."""
    return {g for g in v.gear if v.held(g) > 0}


def _wanted_raw(v: View) -> set[str]:
    return {g for g in RAW if v.held(g) > 0 and v.bid(g) is not None}


def _sells_wanted_raw_met(v: View, a: dict[str, Any]) -> bool:
    return bool(sells(a, _wanted_raw(v)))


def _scarce_raw(v: View) -> set[str]:
    """Raw materials it holds that buyers bid for and nobody offers."""
    return {g for g in _wanted_raw(v) if v.ask(g) is None}


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


def _unsold(v: View) -> set[str]:
    return {p for p in v.recipes() if v.owned(p) >= UNSOLD_FLAG and v.bid(p) is None}


def _produces_possible_met(v: View, a: dict[str, Any]) -> bool:
    if a.get("action") != "produce":
        return True
    product = a.get("target")
    if product not in v.recipes():
        return False
    quantity = int(a.get("quantity") or 1)
    return v.runs(product) >= quantity and v.capacity >= quantity


def _equips_only_held_met(v: View, a: dict[str, Any]) -> bool:
    if a.get("action") != "equip":
        return True
    return not v.away and v.held(a.get("target")) > 0
