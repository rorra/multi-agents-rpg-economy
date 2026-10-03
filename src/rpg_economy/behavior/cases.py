"""The behaviour check's situations: one per rule each role must follow.

A situation puts the engine in a state by hand (a few mechanical cycles, then
holdings and orders set directly) and names the agent asked. It is not scored
by an expectation of its own: every obligation that applies to the agent's
observation (``obligations``) is checked on every answer, so a situation built
to test one rule also catches a break of any other. ``focus`` names the rules
it was built for; a test checks they apply.

Most situations come in variants. ``clean`` holds only what the rule is about;
``noisy`` adds what surrounded real runs' decisions (a potion market in
motion, other gear on offer, a potion of its own, an order of its own resting),
which in the pilot was enough to draw adventurers from selling their raw
materials into trading potions; ``poor`` leaves the agent too little gold to
buy, so the rules that need gold give way to those that raise it. Real moments
from a run log are added by ``replay``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from rpg_economy.domain import Condition, OrderSide, parse_good
from rpg_economy.engine import Engine

#: The conditions an agent situation runs in: the two without a Director differ
#: in the bulletin alone, which changes the prompt.
DECENTRALIZED = (Condition.B, Condition.C)

Build = Callable[[Engine], str]
Modifier = Callable[[Engine, str], None]


@dataclass(frozen=True)
class Situation:
    """One state of the world, the agent asked in it, and the rules it tests."""

    name: str
    #: Builds the state and returns the agent asked (None for a logged moment).
    build: Build | None
    #: The obligations the situation was built for; each must apply in it.
    focus: tuple[str, ...] = ()
    conditions: tuple[Condition, ...] = DECENTRALIZED
    #: A Condition A TASK for the agent: (type, params).
    task: tuple[str, dict] | None = None
    #: The model answers as the Director; the check is a valid decision.
    director: bool = False
    #: Where it comes from: what it reproduces, in a line.
    origin: str = ""


# -- helpers -------------------------------------------------------------------
def _advance(engine: Engine, cycles: int) -> None:
    for _ in range(cycles):
        engine.run_cycle()


def _hold(engine: Engine, agent: str, good: str, quantity: int) -> None:
    account = engine.state.account(agent)
    key = parse_good(good)
    account.inventory[key] = account.inventory.get(key, 0) + quantity


def _order(engine: Engine, agent: str, side: OrderSide, good: str, quantity: int, price: str):
    """A resting order, with the holdings or gold it needs given first."""
    if side is OrderSide.SELL:
        _hold(engine, agent, good, quantity)
    else:
        engine.state.account(agent).gold += Decimal(price) * quantity
    return engine.submit_order(agent, side, parse_good(good), quantity, price)


def _trade(engine: Engine, seller: str, buyer: str, good: str, quantity: int, price: str) -> None:
    _order(engine, seller, OrderSide.SELL, good, quantity, price)
    _order(engine, buyer, OrderSide.BUY, good, quantity, price)


def _set(engine: Engine, agent: str, **holdings: int) -> None:
    engine.state.account(agent).inventory = {parse_good(g): q for g, q in holdings.items()}


def _other(agent: str, *candidates: str) -> str:
    return next(c for c in candidates if c != agent)


# -- variants ------------------------------------------------------------------
def noisy(engine: Engine, agent: str) -> None:
    """The traffic around real runs' decisions, none of it the rule's subject."""
    seller = _other(agent, "alchemy_2", "alchemy_1")
    buyer = _other(agent, "mage_2", "mage_1")
    trader = _other(agent, "ranger_2", "warrior_2")
    _trade(engine, seller, trader, "health_potion", 1, "4.10")
    _order(engine, seller, OrderSide.SELL, "health_potion", 3, "3.90")
    _order(engine, buyer, OrderSide.BUY, "health_potion", 1, "3.60")
    account = engine.state.account(agent)
    if account.role.value == "adventurer":
        _hold(engine, agent, "health_potion", 2)
    else:
        smith = _other(agent, "smithing_2", "smithing_1")
        if not agent.startswith("smithing"):
            _order(engine, smith, OrderSide.SELL, "armor", 1, "46.00")


def resting(engine: Engine, agent: str) -> None:
    """Back from a trip and resting: when gear breaks, and when there is time to buy."""
    engine.state.account(agent).recovering_until_tick = engine.tick + 4


def poor(engine: Engine, agent: str) -> None:
    """Too little gold to buy anything worth having."""
    engine.state.account(agent).gold = Decimal("6")


def _variants(
    base: str,
    build: Build,
    focus: tuple[str, ...],
    origin: str,
    variants: dict[str, tuple[Modifier, ...]],
    focus_of: dict[str, tuple[str, ...]] | None = None,
) -> list[Situation]:
    out = []
    for label, modifiers in variants.items():

        def built(engine: Engine, _build=build, _mods=modifiers) -> str:
            agent = _build(engine)
            for modify in _mods:
                modify(engine, agent)
            return agent

        out.append(
            Situation(
                name=f"{base}:{label}",
                build=built,
                focus=(focus_of or {}).get(label, focus),
                origin=origin,
            )
        )
    return out


CLEAN_NOISY: dict[str, tuple[Modifier, ...]] = {"clean": (), "noisy": (noisy,)}


# -- adventurer states ------------------------------------------------------------
def resting_holds_wanted_raw(engine: Engine) -> str:
    _advance(engine, 20)
    _order(engine, "smithing_1", OrderSide.BUY, "iron", 2, "10.00")
    _order(engine, "woodworking_1", OrderSide.BUY, "wood", 2, "8.00")
    _set(engine, "ranger_1", iron=2, wood=1)
    engine.state.account("ranger_1").recovering_until_tick = engine.tick + 5
    return "ranger_1"


def away_holds_wanted_raw(engine: Engine) -> str:
    _advance(engine, 20)
    _order(engine, "smithing_2", OrderSide.BUY, "leather", 2, "9.50")
    engine.start_expedition("mage_1", "green_hollow")
    _set(engine, "mage_1", leather=3, arcane_crystal=2)
    return "mage_1"


def ready_holds_wanted_raw(engine: Engine) -> str:
    _advance(engine, 20)
    _order(engine, "smithing_1", OrderSide.BUY, "leather", 2, "9.50")
    _set(engine, "warrior_1", leather=3, health_potion=1)
    return "warrior_1"


def scarce_herbs(engine: Engine) -> str:
    """Adventurers sold every herb at the
    alchemists' bid, and the scarce good's price fell from 6 to 5."""
    _advance(engine, 36)
    _trade(engine, "ranger_2", "alchemy_1", "herbs", 1, "6.00")
    _order(engine, "alchemy_1", OrderSide.BUY, "herbs", 3, "6.00")
    _order(engine, "alchemy_2", OrderSide.BUY, "herbs", 2, "6.00")
    _set(engine, "ranger_1", herbs=4)
    engine.state.account("ranger_1").recovering_until_tick = engine.tick + 4
    return "ranger_1"


def raw_on_offer(engine: Engine) -> str:
    """Adventurers bid for arcane crystal on offer "to diversify", which
    the market refuses (11 times in 164 activations)."""
    _advance(engine, 30)
    _trade(engine, "mage_2", "woodworking_1", "arcane_crystal", 1, "15.00")
    _order(engine, "mage_2", OrderSide.SELL, "arcane_crystal", 2, "14.00")
    _order(engine, "ranger_2", OrderSide.SELL, "wood", 2, "8.00")
    _set(engine, "warrior_2", health_potion=1)
    warrior = engine.state.account("warrior_2")
    warrior.gold = Decimal("220")
    warrior.recovering_until_tick = engine.tick + 4
    return "warrior_2"


def broken_weapon_offered(engine: Engine) -> str:
    _advance(engine, 9)
    _order(engine, "smithing_1", OrderSide.SELL, "melee_weapon", 1, "39.00")
    _order(engine, "smithing_2", OrderSide.SELL, "armor", 1, "51.00")
    warrior = engine.state.account("warrior_1")
    warrior.gold = Decimal("140")
    warrior.equipment.clear()
    return "warrior_1"


def broken_weapon_offered_holds_raw(engine: Engine) -> str:
    """With too little gold for the weapon, its raw materials are the way to it."""
    agent = broken_weapon_offered(engine)
    _order(engine, "smithing_2", OrderSide.BUY, "iron", 2, "10.00")
    _set(engine, agent, iron=3)
    return agent


def potions_offered(engine: Engine) -> str:
    """With potions optional and not paying, adventurers bought 3 in 90
    ticks while alchemists offered them all along."""
    _advance(engine, 20)
    _order(engine, "alchemy_1", OrderSide.SELL, "health_potion", 6, "4.50")
    _set(engine, "ranger_1", wood=1)
    engine.state.account("ranger_1").gold = Decimal("120")
    return "ranger_1"


def broken_weapon_unoffered(engine: Engine) -> str:
    _advance(engine, 9)
    warrior = engine.state.account("warrior_1")
    warrior.gold = Decimal("140")
    warrior.equipment.clear()
    return "warrior_1"


def broken_staff_offered(engine: Engine) -> str:
    _advance(engine, 30)
    _order(engine, "woodworking_1", OrderSide.SELL, "staff_focus", 1, "70.00")
    _order(engine, "woodworking_2", OrderSide.SELL, "bow_gear", 1, "33.00")
    mage = engine.state.account("mage_2")
    mage.equipment.clear()
    mage.gold = Decimal("180")
    _set(engine, "mage_2", arcane_crystal=2, health_potion=1)
    return "mage_2"


def holds_potion(engine: Engine) -> str:
    """Resting with the potion its next trip would use, potions offered
    with no bids and an armor bid waiting, adventurers sold the potion "to raise
    cash" (23 offers), undercutting the alchemists."""
    _advance(engine, 9)
    _order(engine, "alchemy_1", OrderSide.SELL, "health_potion", 4, "4.00")
    _order(engine, "mage_1", OrderSide.BUY, "armor", 1, "51.00")
    _set(engine, "ranger_2", health_potion=1, herbs=1)
    engine.state.account("ranger_2").gold = Decimal("74")
    return "ranger_2"


# -- artisan states -----------------------------------------------------------------
def _smith_with_inputs(engine: Engine) -> str:
    _set(engine, "smithing_1", iron=4, leather=4)
    engine.state.account("smithing_1").capacity_credit = Decimal("1")
    return "smithing_1"


def weapons_wanted(engine: Engine) -> str:
    _advance(engine, 50)
    _order(engine, "warrior_1", OrderSide.BUY, "melee_weapon", 1, "40.00")
    _order(engine, "warrior_2", OrderSide.BUY, "melee_weapon", 1, "39.00")
    _order(engine, "smithing_2", OrderSide.SELL, "armor", 2, "51.00")
    return _smith_with_inputs(engine)


def armor_wanted(engine: Engine) -> str:
    _advance(engine, 50)
    _order(engine, "ranger_1", OrderSide.BUY, "armor", 1, "52.00")
    _order(engine, "smithing_2", OrderSide.SELL, "melee_weapon", 2, "39.00")
    return _smith_with_inputs(engine)


def weapon_wanted_no_iron(engine: Engine) -> str:
    """Warriors bid for weapons, smiths without iron sold armor."""
    _advance(engine, 50)
    _order(engine, "warrior_1", OrderSide.BUY, "melee_weapon", 1, "46.00")
    _order(engine, "smithing_1", OrderSide.SELL, "armor", 1, "44.00")
    _set(engine, "smithing_1", leather=2, armor=1)
    engine.state.account("smithing_1").gold = Decimal("130")
    return "smithing_1"


def weapon_wanted_iron_offered(engine: Engine) -> str:
    agent = weapon_wanted_no_iron(engine)
    _order(engine, "warrior_2", OrderSide.SELL, "iron", 2, "12.00")
    return agent


def holds_wanted_weapon(engine: Engine) -> str:
    _advance(engine, 40)
    _order(engine, "warrior_2", OrderSide.BUY, "melee_weapon", 1, "46.00")
    _set(engine, "smithing_2", melee_weapon=1, leather=1)
    return "smithing_2"


def weapon_already_offered(engine: Engine) -> str:
    """Artisans re-offered goods already on their own sell orders."""
    _advance(engine, 40)
    _set(engine, "smithing_2", armor=0)
    _order(engine, "smithing_2", OrderSide.SELL, "armor", 2, "52.00")
    _order(engine, "ranger_1", OrderSide.BUY, "armor", 1, "47.00")
    return "smithing_2"


def staff_wanted_bow_lowball(engine: Engine) -> str:
    """A woodworker held a staff the mages bid 76 for and never
    offered it, and sold bows worth about 34 in inputs to a 15.50 bid."""
    _advance(engine, 50)
    _trade(engine, "ranger_1", "woodworking_2", "wood", 1, "9.90")
    _trade(engine, "mage_1", "woodworking_2", "leather", 1, "9.95")
    woodworker = engine.state.account("woodworking_1")
    woodworker.gold += Decimal("100")
    _order(engine, "woodworking_1", OrderSide.SELL, "bow_gear", 1, "46.00")
    _advance(engine, 7)  # its bow offer expires unsold
    _order(engine, "ranger_2", OrderSide.BUY, "bow_gear", 1, "15.50")
    _order(engine, "mage_2", OrderSide.BUY, "staff_focus", 1, "76.00")
    _set(engine, "woodworking_1", bow_gear=2, staff_focus=1, leather=1, wood=1)
    return "woodworking_1"


def bows_no_market(engine: Engine) -> str:
    """Pilot: with no market yet, a woodworker sold bows at 9 (inputs about 34)."""
    _advance(engine, 9)
    _set(engine, "woodworking_1", bow_gear=2)
    return "woodworking_1"


def armor_unsold_no_inputs(engine: Engine) -> str:
    """A smith with unsold armor, no inputs and no bids cut below cost."""
    _advance(engine, 60)
    _order(engine, "smithing_2", OrderSide.SELL, "armor", 1, "41.00")
    _set(engine, "smithing_1", armor=3)
    return "smithing_1"


def potion_glut(engine: Engine) -> str:
    """Pilot: alchemists kept brewing while 20+ potions sat unsold with no bid."""
    _advance(engine, 100)
    _order(engine, "alchemy_2", OrderSide.SELL, "health_potion", 4, "4.40")
    _set(engine, "alchemy_1", health_potion=25, herbs=3)
    engine.state.account("alchemy_1").gold = Decimal("60")
    return "alchemy_1"


def herb_bids_unfilled(engine: Engine) -> str:
    """Pilot: alchemists re-bid the starting 6.00 for herbs after their bids
    went unfilled, so the scarce good's price never moved."""
    _advance(engine, 40)
    _trade(engine, "ranger_1", "alchemy_2", "herbs", 1, "6.00")
    _order(engine, "alchemy_1", OrderSide.BUY, "herbs", 3, "6.00")
    _advance(engine, 7)
    _order(engine, "alchemy_1", OrderSide.BUY, "herbs", 3, "6.00")
    _advance(engine, 7)
    _order(engine, "alchemy_2", OrderSide.BUY, "herbs", 3, "6.00")
    _set(engine, "alchemy_1", arcane_crystal=2)
    engine.state.account("alchemy_1").gold = Decimal("140")
    return "alchemy_1"


def herbs_offered_with_task(engine: Engine) -> str:
    """Pilot (A): two of three ANALYZE_MARKET tasks ended in a wait."""
    _advance(engine, 20)
    _order(engine, "ranger_1", OrderSide.SELL, "herbs", 3, "6.50")
    _set(engine, "alchemy_1", arcane_crystal=2)
    engine.state.account("alchemy_1").gold = Decimal("120")
    return "alchemy_1"


def director_cycle(engine: Engine) -> str:
    """Pilot (A): 14% of the Director's outputs were invalid."""
    _advance(engine, 20)
    _order(engine, "alchemy_1", OrderSide.BUY, "herbs", 3, "6.50")
    _order(engine, "warrior_1", OrderSide.SELL, "iron", 2, "10.00")
    return "director"


PILOT_RAW = (
    "adventurers holding raw materials buyers bid for sold them 18-25% of the time"
)
PILOT_GEAR = (
    "adventurers without their required item waited or traded instead of buying it"
)

SITUATIONS: list[Situation] = [
    # adventurers
    *_variants(
        "adv_resting_holds_wanted_raw",
        resting_holds_wanted_raw,
        ("adv_sells_wanted_raw", "adv_no_set_out_when_unable"),
        PILOT_RAW,
        CLEAN_NOISY,
    ),
    *_variants(
        "adv_scarce_herbs",
        scarce_herbs,
        ("adv_sells_wanted_raw", "adv_asks_above_bid_when_scarce"),
        "during the herb shock, every herb sold at the bid and the price fell",
        CLEAN_NOISY,
    ),
    *_variants(
        "adv_raw_on_offer",
        raw_on_offer,
        ("adv_no_raw_buy",),
        "adventurers bid for arcane crystal on offer, which the market refuses",
        CLEAN_NOISY,
    ),
    *_variants(
        "adv_away_holds_wanted_raw",
        away_holds_wanted_raw,
        ("adv_sells_wanted_raw",),
        PILOT_RAW,
        CLEAN_NOISY,
    ),
    *_variants(
        "adv_ready_holds_wanted_raw",
        ready_holds_wanted_raw,
        ("adv_ready_acts",),
        "adventurers able to set out waited or traded potions",
        CLEAN_NOISY,
    ),
    *_variants(
        "adv_broken_weapon_offered",
        broken_weapon_offered,
        ("adv_buys_paying_gear",),
        PILOT_GEAR,
        {"resting": (resting,), "resting_noisy": (noisy, resting), "ready": ()},
        focus_of={"ready": ("adv_ready_acts",)},
    ),
    *_variants(
        "adv_broken_weapon_offered_holds_raw",
        broken_weapon_offered_holds_raw,
        ("adv_buys_paying_gear", "adv_sells_wanted_raw"),
        PILOT_GEAR,
        {"rich": (resting,), "poor": (poor,)},
        focus_of={"poor": ("adv_ready_acts", "buys_within_gold")},
    ),
    *_variants(
        "adv_potions_offered",
        potions_offered,
        ("adv_buys_paying_gear",),
        "adventurers bought 3 potions in 90 ticks",
        # The noisy variant would hand it potions of its own.
        {"resting": (resting,), "ready": ()},
        focus_of={"ready": ("adv_ready_acts",)},
    ),
    *_variants(
        "adv_holds_potion",
        holds_potion,
        ("adv_keeps_gear",),
        "adventurers sold the potions their next trip would use",
        {"resting": (resting,), "resting_noisy": (noisy, resting), "poor": (resting, poor)},
    ),
    *_variants(
        "adv_broken_weapon_unoffered",
        broken_weapon_unoffered,
        ("adv_bids_for_paying_gear",),
        PILOT_GEAR,
        {"resting": (resting,), "resting_noisy": (noisy, resting)},
    ),
    *_variants(
        "adv_broken_staff_offered",
        broken_staff_offered,
        ("adv_buys_paying_gear", "adv_buys_only_its_gear"),
        "mages never got a staff",
        {"resting": (resting,)},
    ),
    # artisans
    *_variants(
        "art_weapons_wanted",
        weapons_wanted,
        ("art_crafts_wanted",),
        "pilot: smiths crafted armor, which sat unsold, while warriors bid for weapons",
        CLEAN_NOISY,
    ),
    *_variants(
        "art_armor_wanted",
        armor_wanted,
        ("art_crafts_wanted",),
        "the mirror of art_weapons_wanted: the choice follows demand, not list order",
        {"clean": ()},
    ),
    *_variants(
        "art_weapon_wanted_no_iron",
        weapon_wanted_no_iron,
        ("art_buys_input_for_wanted",),
        "smiths without iron kept offering armor",
        CLEAN_NOISY,
    ),
    *_variants(
        "art_weapon_wanted_iron_offered",
        weapon_wanted_iron_offered,
        ("art_buys_input_for_wanted",),
        "as above, with iron on offer",
        {"clean": ()},
    ),
    *_variants(
        "art_holds_wanted_weapon",
        holds_wanted_weapon,
        ("art_sells_wanted_stock",),
        "artisans held products buyers bid for (sold 4-13% of the time)",
        CLEAN_NOISY,
    ),
    *_variants(
        "art_stock_already_offered",
        weapon_already_offered,
        ("sells_within_stock", "art_no_sale_below_cost"),
        "artisans offered goods already on their own sell orders",
        {"clean": ()},
    ),
    *_variants(
        "art_staff_wanted_bow_lowball",
        staff_wanted_bow_lowball,
        ("art_sells_wanted_stock", "art_no_sale_below_cost"),
        "staff unsold to a 76 bid, bows sold at 15.50",
        {"clean": ()},
    ),
    *_variants(
        "art_bows_no_market",
        bows_no_market,
        ("art_no_sale_below_cost",),
        "pilot: with no market yet, bows sold at 9 (inputs about 34)",
        {"clean": ()},
    ),
    *_variants(
        "art_armor_unsold_no_inputs",
        armor_unsold_no_inputs,
        ("art_stops_unsold", "art_no_sale_below_cost"),
        "unsold armor cut below cost",
        {"clean": ()},
    ),
    *_variants(
        "art_potion_glut",
        potion_glut,
        ("art_stops_unsold",),
        "pilot: alchemists kept brewing while 20+ potions sat unsold",
        {"clean": ()},
    ),
    *_variants(
        "art_herb_bids_unfilled",
        herb_bids_unfilled,
        ("art_raises_failed_bid",),
        "pilot: alchemists re-bid the starting price after their bids went unfilled",
        CLEAN_NOISY,
    ),
    # Condition A: the same rules under a TASK
    Situation(
        name="task_analyze_market_herbs",
        build=herbs_offered_with_task,
        focus=("task_answered_with_action",),
        conditions=(Condition.A,),
        task=("ANALYZE_MARKET", {"good": "herbs"}),
        origin="pilot (A): two of three ANALYZE_MARKET tasks ended in a wait",
    ),
    Situation(
        name="task_evaluate_production_weapons_wanted",
        build=weapons_wanted,
        focus=("task_answered_with_action", "art_crafts_wanted"),
        conditions=(Condition.A,),
        task=("EVALUATE_PRODUCTION", {"product": "melee_weapon"}),
        origin="the smiths' rule under the Director",
    ),
    Situation(
        name="task_procurement_no_iron",
        build=weapon_wanted_no_iron,
        focus=("task_answered_with_action", "art_buys_input_for_wanted"),
        conditions=(Condition.A,),
        task=("MANAGE_PROCUREMENT", {"good": "iron"}),
        origin="the missing-input rule under the Director",
    ),
    Situation(
        name="task_inventory_resting_raw",
        build=resting_holds_wanted_raw,
        focus=("task_answered_with_action", "adv_sells_wanted_raw"),
        conditions=(Condition.A,),
        task=("MANAGE_INVENTORY", {"good": "iron"}),
        origin="the raw-material rule under the Director",
    ),
    Situation(
        name="task_expedition_broken_weapon",
        build=broken_weapon_offered,
        focus=("adv_ready_acts",),
        conditions=(Condition.A,),
        task=("EVALUATE_EXPEDITION", {"resource": "iron", "zone": "green_hollow"}),
        origin="the gear rule under the Director",
    ),
    Situation(
        name="director_answers_validly",
        build=director_cycle,
        conditions=(Condition.A,),
        director=True,
        origin="pilot (A): 14% of the Director's outputs were invalid",
    ),
]

#: The Director's answers are judged on validity alone.
DIRECTOR_THRESHOLD = 0.85
