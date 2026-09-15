"""Prompt construction, versioned and hashed (design v1.0 s15).

Prompts are assembled from labelled sections. That is not presentation: the
token buckets of s8.4 require knowing how much of a prompt was peer-to-peer
message content and how much was bulletin, and a provider reports only one input
total per call. Sections give the proportions that split it.

Every template is versioned and hashed into run metadata, because results can
be sensitive to the wording and the design requires prompts be fixed before the
final runs.

Traits (s7) enter here and nowhere else. They shape how an agent reasons; they
never reach engine validation, which sees only the resulting action.
"""

from __future__ import annotations

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import ActionType, Condition, ObservationSignal, Role

#: Bump when any template below changes. Recorded in run metadata.
PROMPT_VERSION = "2.61"

#: Section names. ``p2p`` and ``bulletin`` map onto token buckets of the same
#: meaning; everything else is charged to the agent.
SECTION_SYSTEM = "system"
SECTION_STATE = "state"
SECTION_MARKET = "market"
SECTION_DIRECTORY = "directory"
SECTION_TASK = "task"
SECTION_P2P = "p2p"
SECTION_BULLETIN = "bulletin"
SECTION_INSTRUCTIONS = "instructions"

#: Heads the still-open proposals block of the p2p section.
OPEN_PROPOSALS_HEADER = "Proposals still open"


ROLE_BRIEF: dict[Role, str] = {
    Role.ADVENTURER: (
        "You are an Adventurer. You gather raw materials on expeditions, maintain "
        "equipment that wears out, and sell what you bring back. Zones differ in "
        "risk: higher risk means a larger expected haul (see Zones). "
        "gear_if_carried lists every item your subclass can use and what it adds to "
        "your chance of success while you carry it: it is not what you hold. What "
        "you carry is your equipment (worn, with durability left) and the potions "
        "in your inventory. your_success_chance is your chance in each zone if you set out "
        "now, with the gear you wear and the potions you hold. Each item also adds "
        "to the gold a successful trip brings back (see Gear gold). You may set out "
        "with or without gear: you start with your main item equipped, and once it "
        "breaks you may buy another. Weapons and armor count while equipped, and "
        "setting out equips any you hold in place of a worn-out one, so a bought "
        "item needs no equip action. Each expedition wears "
        "them down (your equipment shows each item's durability points left; the "
        "loss per expedition is under Equipment wear). Potions "
        "count when you hold them as you set out, and setting out consumes the "
        "potions the trip uses. After "
        "each expedition you rest a few ticks before you can set out again "
        "(can_set_out_from_tick); you can trade meanwhile. While away on an "
        "expedition you can trade (your orders stay on the book and you may place "
        "or cancel orders) but you cannot equip or set out. You "
        "cannot craft: raw materials are worth to you only what they sell for, so "
        "you sell them and never buy them (the market refuses an adventurer's bid "
        "for a raw material). Gold is what buys your equipment and potions: you buy "
        "them for your trips and never sell them (the market refuses an adventurer's "
        "offer of a product)."
    ),
    Role.ARTISAN: (
        "You are an Artisan. You buy raw materials, craft products of your "
        "specialization within a per-cycle capacity counted in recipe runs, and "
        "sell what you make. You cannot craft outside your specialization, and you "
        "do not go on expeditions: only adventurers gather raw materials. Gold "
        "comes to you only from selling what you craft above what its inputs cost: a "
        "sale at or below that cost earns nothing. A product you hold pays a small "
        "storage fee every tick until it is sold (see Costs), far less than the margin "
        "of a sale at a fair price, so hold out for one. "
        "Buying inputs, crafting and offering the products each take an "
        "activation of their own, so one unit at a time leaves bids unserved: "
        "while buyers keep bidding, buy inputs for several runs in one "
        "place_orders, produce several runs at once and offer them together."
    ),
}

RULES = (
    "Rules of the world:\n"
    "- You may attempt exactly ONE action per activation.\n"
    "- The engine validates every action against inventory, gold, recipes and "
    "capacity. An invalid action is discarded, not partially applied, except "
    "place_orders, whose orders are each judged on their own.\n"
    "- Orders are limit orders on a shared book with price-time priority. They "
    "expire after a fixed number of ticks.\n"
    "- You cannot sell what you do not hold, or buy without the gold to pay.\n"
    "- available_gold leaves out gold your open orders reserve, frozen gold, and "
    "wealth tax you owe (tax_owed); available_inventory leaves out what your open "
    "orders have reserved. Only the available part can be spent, sold, used or "
    "equipped.\n"
    "- Tax owed is collected from free gold (see Costs); until then that gold "
    "cannot be spent or bid.\n"
    "- There is no official price: trades set prices. The price guide says where "
    "prices started, and a typical price, when shown, where a good has usually "
    "traded: a reference for what it is worth, not a limit. Prices follow supply "
    "and demand. When a good is scarce (few or no offers, bids left unfilled) "
    "buyers pay more and sellers ask more, above the reference if that is what it "
    "takes to trade; when it piles up unsold, prices fall, below the reference if "
    "need be. Price from what the market shows now, with the reference in mind.\n"
    "- An order does not reprice itself: it rests at its price until it fills, is "
    "cancelled or expires.\n"
    "- Unsold stock ties up gold, and for artisans it also costs "
    "storage (see Costs).\n"
    # The observability rule of design v1.0 s5.2, which s7 requires the prompt
    # to state. Descriptive only: it says what peers can see, not what to do
    # about it. True in every condition, so the world rules stay identical;
    # how a need can be made known (a message in B and C, the RESULT fields to
    # the Director in A) is stated with each condition's action schema.
    "- Peers cannot see your needs, holdings or plans. The public directory "
    "shows only each agent's id, role and specialization."
)

_ACTION_LINES = (
    "- buy/sell need target, quantity and a positive price.\n"
    "- produce needs target (a product) and quantity: a number of recipe runs. "
    "Each run consumes the recipe's inputs and yields the recipe's output "
    "quantity; production_capacity_left counts runs.\n"
    "- start_expedition needs target (a zone id).\n"
    "- cancel_order needs order_id: the id of one of your_open_orders.\n"
    "- equip needs target (a product in your available_inventory: equipping does "
    "not buy it). Only adventurers equip, and not while away on an expedition.\n"
    "- place_orders places several market orders in one action: orders is a list "
    'of {"action": "buy"|"sell"|"cancel_order", "target", "quantity", "price", '
    '"order_id"}, '
    "each filled in as that action needs, up to the limit under Costs. The engine "
    "takes each order on its own: those it accepts go on the book, the others are "
    "refused without affecting the rest. The top-level target, quantity and price "
    "are then null.\n"
)
_SHORT_REASON = (
    "short_reason is one short sentence, recorded for analysis only. It never "
    "affects what the engine does."
)

#: What each message means. A message moves no gold and no goods, and an
#: agent that is not told so can take an ACK for a closed sale.
_MESSAGE_LINES = (
    "- Peers cannot see your needs unless you tell them in a message.\n"
    "- Messages move no gold and no goods: a trade happens only when orders "
    "cross on the book.\n"
    "- REQUEST asks for a good, OFFER proposes one, INFORM shares a market "
    "signal.\n"
    "- ACK accepts a REQUEST or OFFER you received and, in the same action, places "
    "your matching order: a buy when you accept an OFFER, a sell when you accept a "
    "REQUEST, at the quantity and price you give. If that order cannot be placed, "
    "the ACK is not sent. REJECT declines. Each answers one message from that agent "
    "about that good, once. ACK and REJECT are never answered: the engine refuses "
    "an ACK or REJECT sent to anything else.\n"
    "- At most one REQUEST or OFFER from one agent to another about a good is open "
    "at a time: a new one replaces the open one, and an ACK or REJECT answers the "
    "latest.\n"
    "- A REQUEST or OFFER can be answered only while it is open: it expires after "
    "the same number of ticks as an order (see Costs).\n"
    "- An ACK to your REQUEST or OFFER means the other agent placed its matching "
    "order; a REJECT means it declined. Neither places an order for you.\n"
    "- A new message to you is shown at your next activation. A REQUEST or OFFER "
    "you have not answered stays listed after that, among the open proposals, "
    "with its ticks_left, until you answer it or it expires.\n"
)

#: Conditions B and C: peer messages exist, RESULT fields do not.
ACTION_SCHEMA = (
    "Reply with a single JSON object and nothing else:\n"
    '{"action": <action>, "target": <string|null>, "quantity": <integer|null>, '
    '"price": <number|null>, "recipient": <agent_id|null>, '
    '"priority": "high"|"medium"|"low", "short_reason": <string>, '
    '"message_type": <REQUEST|OFFER|INFORM|REJECT|ACK|null>, "order_id": <integer|null>, '
    '"orders": [...]|null}\n'
    "Valid actions: "
    + ", ".join(a.value for a in ActionType)
    + ".\n"
    + _ACTION_LINES
    + "- send_message needs recipient (an agent id), message_type and target (the "
    "good the message is about). REQUEST, OFFER and ACK also need quantity and "
    "price.\n"
    "- message_type is null for every other action.\n"
    + _MESSAGE_LINES
    + "- wait needs nothing.\n"
    + _SHORT_REASON
)

#: Condition A: no peer messages (design s3.1); the RESULT fields carry needs,
#: offers and observations to the Director instead. Stating a message action
#: here would only invite invalid output.
ACTION_SCHEMA_A = (
    "Reply with a single JSON object and nothing else:\n"
    '{"action": <action>, "target": <string|null>, "quantity": <integer|null>, '
    '"price": <number|null>, "priority": "high"|"medium"|"low", '
    '"short_reason": <string>, "order_id": <integer|null>, "orders": [...]|null, '
    '"needs": [...], "offers": [...], '
    '"observations": [...]}\n'
    "Valid actions: "
    + ", ".join(a.value for a in ActionType if a is not ActionType.SEND_MESSAGE)
    + ".\n"
    + _ACTION_LINES
    + "- wait needs nothing.\n"
    + _SHORT_REASON
    + "\nYou cannot message other agents. Besides your action, your reply reaches the "
    "Director only through these structured fields (never short_reason), each "
    "optional and empty if you have nothing to report:\n"
    '- "needs": [{"good": <good>, "quantity": <integer>, "priority": "high"|"medium"|"low"}]\n'
    '- "offers": [{"good": <good>, "quantity": <integer>, "price": <positive number|null>}]\n'
    '- "observations": [{"signal": <signal>, "good": <good>}], where signal is one of '
    + ", ".join(s.value for s in ObservationSignal)
    + "."
)


def action_schema(condition: Condition) -> str:
    return ACTION_SCHEMA_A if condition is Condition.A else ACTION_SCHEMA


def _zone_mix(zone) -> str:
    """What a zone yields, most first: "mostly herbs (45%) and leather (21%), also ..."."""
    rates = sorted(
        ((r.value, rate) for r, rate in zone.spawn_per_tick.items() if rate > 0),
        key=lambda kv: (-kv[1], kv[0]),
    )
    total = sum(rate for _, rate in rates)
    share = [(name, round(rate / total * 100)) for name, rate in rates]
    main = " and ".join(f"{name} ({pct}%)" for name, pct in share[:2])
    rest = ", ".join(name for name, _ in share[2:])
    return f"mostly {main}" + (f", also {rest}" if rest else "")


def render_world(config: SimulationConfig, role: Role | None = None) -> str:
    """The fixed rules of production and gathering, identical in every condition.

    Part of the world's rules, not market information: every agent may know
    what a recipe needs and what a zone yields, while prices, stocks and other
    agents' holdings stay private (s5.2).
    """
    recipes = "; ".join(
        f"{product.value} ({recipe.specialization.value}): "
        + " + ".join(
            f"{units} {resource.value}"
            for resource, units in sorted(recipe.inputs.items(), key=lambda kv: kv[0].value)
        )
        + f" -> {recipe.output_quantity}"
        for product, recipe in sorted(config.economy.recipes.items(), key=lambda kv: kv[0].value)
    )
    expeditions = config.expeditions
    zones = "; ".join(
        f"{zone.id}: risk {zone.risk.value}, {expeditions.duration_ticks[zone.risk]} ticks, "
        f"a success carries up to {expeditions.expected_yield[zone.risk]} units, as many as "
        f"the zone holds, and "
        f"{expeditions.gold_loot.get(zone.risk, 0):g} gold, " + _zone_mix(zone)
        for zone in expeditions.zones
    )
    if expeditions.crowding is not None:
        # Told only each zone's risk, adventurers kept to the safe one.
        zones += (
            f". A zone's resources are limited: it holds what appears in it over at most "
            f"{expeditions.pool_capacity_ticks} ticks, and trips that arrive together share "
            "it, so a crowded zone's trips bring back less and one nobody visits fills up. "
            "Its gold is limited the same way, holding what appears over at most "
            f"{expeditions.gold_capacity_ticks or expeditions.pool_capacity_ticks} ticks: a "
            "success brings its full gold only while the zone holds it, and successes that "
            "arrive together share what it holds. "
            "A zone that got more than an even share of the trips set out over the last "
            f"{expeditions.crowding.window_ticks} ticks is also harder: "
            f"{expeditions.crowding.per_trip:g} less success for each trip over that share, "
            f"up to {expeditions.crowding.cap:g}; one that got fewer is as much easier"
        )
    failure = (
        f"a failed expedition brings back about "
        f"{float(expeditions.partial_yield_on_failure) * 100:g}% of the units and no gold"
    )
    loot = ", ".join(
        f"{subclass.value} x{settings.loot_multiplier:g}"
        for subclass, settings in sorted(expeditions.subclasses.items(), key=lambda kv: kv[0].value)
        if settings.loot_multiplier != 1
    )
    if loot:
        failure += f"; gold found is multiplied by subclass: {loot}"
    economy = config.economy
    taxes = economy.taxes
    storage = ", ".join(
        f"{role.value} {cost} gold"
        for role, cost in sorted(economy.storage_cost_per_unit.items(), key=lambda kv: kv[0].value)
    )
    # Storage and owed tax are charged as each macro-cycle opens. Agents reason
    # in ticks, so the period is stated in ticks: every tick when a cycle is one.
    ticks_per_cycle = config.time.ticks_per_cycle
    period = "every tick" if ticks_per_cycle == 1 else f"every {ticks_per_cycle} ticks"
    if economy.floor_at_initial_gold:
        starting = ", ".join(
            f"{role} {max(gold, economy.charge_floor).normalize():f}"
            for role, gold in sorted(economy.initial_gold.items())
        )
        floor_clause = (
            f"storage and the profit tax never take your gold below your starting gold ({starting})"
        )
    else:
        floor_clause = f"storage never takes your gold below {economy.charge_floor:g}"
    storage_clause = (
        f"storage, charged at the start of {period} per unit of goods held: {storage}; "
        f"{floor_clause}"
        if storage
        else "storage: none"
    )
    costs = (
        f"Costs: orders expire after {config.market.order_expiry_ticks} ticks; "
        f"one place_orders carries at most {config.market.max_orders_per_activation} orders "
        "(there is no limit on how many orders you have open); "
        f"profit tax, paid by the seller of a finished product: "
        f"{taxes.profit_rate * 100:g}% of what the sale brings in above the cost of its "
        "inputs at their last traded prices, nothing on a sale at or below that cost; "
        f"wealth tax: every {taxes.wealth_every_ticks} ticks, {taxes.wealth_rate * 100:g}% "
        f"of gold above {taxes.wealth_exemption:g}, frozen gold and tax already owed "
        f"not taxed; tax that cannot be paid is owed (tax_owed), collected from free "
        f"gold at the start of {period}, and cannot be spent or bid meanwhile; "
        f"{storage_clause}"
    )
    # A failed trip rests rest_ticks plus its risk's recovery; a zero
    # recovery adds nothing and is not stated.
    extra = [
        f"{risk.value} +{ticks}"
        for risk, ticks in sorted(
            expeditions.recovery_ticks_on_failure.items(), key=lambda kv: kv[0].value
        )
        if ticks > 0
    ]
    rest = (
        f"Adventurers rest {expeditions.rest_ticks} ticks after each expedition before "
        "setting out again"
        + (f" (after a failed one, more by zone risk: {', '.join(extra)} ticks)" if extra else "")
    )
    loss = expeditions.durability_loss_per_expedition
    wear = (
        "Equipment wear: none; equipped items do not wear out"
        if loss == 0
        else f"Equipment wear: each expedition takes {loss} durability "
        f"point{'' if loss == 1 else 's'} from every equipped item; an item whose "
        "durability reaches 0 breaks and is gone"
    )
    bonus = expeditions.gear_gold_bonus
    if bonus:
        wear += (
            "\nGear gold: a successful trip's gold is raised, for each item carried, by "
            + ", ".join(
                f"{product.value} +{share * 100:.0f}%"
                for product, share in sorted(bonus.items(), key=lambda kv: kv[0].value)
            )
        )
    own = (
        ""
        if config.market.allow_self_trade
        else "\nOwn orders: an order that would cross an open order of the same agent on "
        "the other side of the same good is refused; that agent cancels its own order "
        "first. place_orders runs its orders in the order listed, so a cancel_order "
        "listed first clears the way"
    )
    # Without any anchor a model invents prices (a bow at 1.2 gold when its
    # inputs are worth 34): what each good sold for before the market opened.
    guide = (
        "Price guide (what each good sold for before this market opened; not an official price, trades set prices from here): "
        + ", ".join(
            f"{good} {price}"
            for good, price in sorted(config.economy.initial_reference_price.items())
        )
    )
    # By role: an artisan never goes on an expedition or wears equipment, so
    # the zones, the rest and the wear are only noise to it. The same text
    # reaches a role in A, B and C; the Director (no role) reads everything.
    gathering = "" if role is Role.ARTISAN else f"Zones: {zones}; {failure}\n{rest}\n{wear}\n"
    limit = config.economy.craft_stock_limit
    stock_rule = (
        f"; an artisan holding {limit} or more units of a product, its own sell orders "
        "included, cannot craft more of it until some sell"
        if limit is not None
        else ""
    )
    return (
        f"Recipes (product (who crafts): inputs per run -> units made per run): "
        f"{recipes}{stock_rule}\n"
        f"{guide}\n"
        f"{gathering}{costs}{own}"
    )
