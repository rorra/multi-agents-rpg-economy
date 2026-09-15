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

import functools
import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal

from rpg_economy.agents.observation import Observation
from rpg_economy.config.models import SimulationConfig, TraitsConfig
from rpg_economy.domain import ActionType, Condition, ObservationSignal, Role
from rpg_economy.logging_.metadata import clear_hash_caches, register_hash_cache

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


def describe_traits(traits: TraitsConfig) -> str:
    """Render the four trait dimensions as guidance (design v1.0 s7)."""

    def band(value: object, low: str, mid: str, high: str) -> str:
        v = float(value)
        return low if v < 0.34 else (mid if v < 0.67 else high)

    return (
        "Your disposition:\n"
        f"- Risk tolerance {traits.risk_tolerance}: you are "
        + band(traits.risk_tolerance, "cautious", "balanced", "willing to gamble")
        + ".\n"
        f"- Time orientation {traits.time_orientation}: you weigh "
        + band(
            traits.time_orientation,
            "immediate payoff over later gain",
            "near and far outcomes about equally",
            "long-run position over immediate payoff",
        )
        + ".\n"
        f"- Commercial aggressiveness {traits.commercial_aggressiveness}: you price "
        + band(traits.commercial_aggressiveness, "conservatively", "moderately", "assertively")
        + ".\n"
        f"- Adaptability {traits.adaptability}: you revise plans "
        + band(traits.adaptability, "slowly", "readily", "quickly")
        + " when conditions change."
    )


@dataclass
class Prompt:
    """A prompt assembled from labelled sections."""

    system: str
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def user(self) -> str:
        """The user message: every non-system section, in a fixed order."""
        order = (
            SECTION_TASK,
            SECTION_STATE,
            SECTION_MARKET,
            SECTION_DIRECTORY,
            SECTION_P2P,
            SECTION_BULLETIN,
            SECTION_INSTRUCTIONS,
        )
        return "\n\n".join(self.sections[name] for name in order if self.sections.get(name))

    def section_sizes(self) -> dict[str, int]:
        """Deterministic size of each section, for token apportionment."""
        from rpg_economy.llm.tokens import estimate_tokens

        sizes = {name: estimate_tokens(text) for name, text in self.sections.items() if text}
        sizes[SECTION_SYSTEM] = estimate_tokens(self.system)
        return sizes


def render_directory(entries: list[dict[str, str | None]]) -> str:
    """The static public directory, one compact record per agent.

    Carries exactly what design v1.0 s6.1 permits -- id, role, specialization,
    nothing else -- in the compact form of IMPLEMENTATION.md section 4.8.1,
    which avoids repeating key names for every agent on every call.
    """
    if not entries:
        return "Other agents: none."
    rows = " ".join(
        f"{entry['agent_id']}:{(entry['role'] or '')}/{entry['specialization'] or '-'}"
        for entry in entries
    )
    return "Other agents (id:role/specialization, all you may know of them):\n" + rows


def set_out_status(observation: Observation) -> str | None:
    """Whether an adventurer can set out now and what gear that pays it lacks.

    Spelled out because a model told the parts (rest, inventory) still set out
    while resting. The gear: shown only in the "Gear" lines, gear that paid for
    itself was bought 1 time in 88, as "ready now" alone led straight to setting
    out; named here, 75-100% of the time while resting (2.34).
    """
    if observation.role != Role.ADVENTURER.value:
        return None
    gear = _paying_gear(observation)
    if observation.is_away:
        status = "not now: you are away on an expedition; you can still trade"
        return f"{status}; meanwhile, {gear}" if gear else status
    if observation.can_set_out_from_tick is not None:
        status = (
            f"not now: you are resting until tick {observation.can_set_out_from_tick} "
            f"(it is tick {observation.tick}); trade meanwhile"
        )
        return f"{status}; meanwhile, {gear}" if gear else status
    # Told "you lack gear that pays: ... bid about 48 for one",
    # warriors set out without a weapon in 17 answers of 20.
    return (
        f"ready now, but you do not carry gear that pays for itself: {gear}. Set out "
        "now and you go without it; a bid stays on the book while you are away, so "
        "bid first and set out next"
        if gear
        else "ready now"
    )


def _paying_gear(observation: Observation) -> str | None:
    """The gear the adventurer lacks that would add more to a trip than it costs."""
    parts = []
    for fact in observation.gear:
        if not fact["pays"]:
            continue
        gains = [Decimal(g) for g in fact["gains"].values()]
        adds = f"adds about {min(gains):.0f}-{max(gains):.0f} a trip"
        item = fact["item"]
        if fact["ask"] is not None:
            if observation.available_gold >= Decimal(fact["ask"]):
                parts.append(f"{item} offered at {fact['ask']} {adds} for {fact['cost_per_trip']}")
        else:
            # 2.35: told "adds ... for 9.75" (the cost a trip), warriors bid 9.75
            # for a weapon worth 39; the price to bid is the item's own.
            lasts = f", lasts {fact['durability']} trips" if fact["durability"] else ""
            parts.append(
                f"nobody offers {'an' if item[0] in 'aeiou' else 'a'} {item} ({adds}{lasts}): "
                f"bid about {fact['price']} for one "
                "(artisans craft what buyers bid for)"
            )
    if any(" for one " in part for part in parts) and any(
        line.startswith("Wanted:") for line in observation.status
    ):
        # 2.35: holding raw materials to sell as well, it sold them and never
        # bid (15%); the two fit in one place_orders.
        parts.append("one place_orders can carry that bid and your raw-material sales")
    return "; ".join(parts) or None


def _json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def build_agent_prompt(
    observation: Observation,
    config: SimulationConfig,
    *,
    condition: Condition = Condition.B,
    task: dict | None = None,
) -> Prompt:
    """Build one activation's prompt from what the agent may see.

    ``task`` is the Director's TASK in Condition A. Its tokens belong to the
    agent bucket: it is the agent's own context, not peer traffic and not the
    bulletin.
    """
    role = Role(observation.role)

    system = "\n\n".join(
        [
            ROLE_BRIEF[role],
            describe_traits(observation.traits),
            RULES,
            render_world(config, role),
        ]
    )

    status = set_out_status(observation)
    headline = (f"Setting out: {status}\n" if status else "") + "".join(
        f"{line}\n" for line in observation.status
    )
    state = (
        "Your state:\n"
        + headline
        + _json(
            {
                "agent_id": observation.agent_id,
                "specialization": observation.specialization,
                "tick": observation.tick,
                "gold": observation.gold,
                "available_gold": observation.available_gold,
                "tax_owed": observation.tax_owed,
                "inventory": observation.inventory,
                "available_inventory": observation.available_inventory,
                "equipment": observation.equipment,
                # Named expedition_gear, the table read as gear held;
                # warriors whose weapon broke set out "with current gear".
                "gear_if_carried": observation.expedition_gear or None,
                "your_success_chance": observation.success_chance or None,
                "away_on_expedition": observation.is_away,
                "can_set_out_from_tick": observation.can_set_out_from_tick,
                "production_capacity_left": observation.production_capacity,
                "your_open_orders": observation.open_orders,
                "your_recent_trades": observation.own_trades or None,
                "your_recent_actions": observation.history or None,
            }
        )
    )

    # With gold and nothing to sell, adventurers bid for raw materials
    # on offer; the rule in the role brief alone held 42-93% of the time, and
    # stated here, where the offers are read, 92-99%.
    follow = (
        "Market, for the goods you follow (raw materials: you only sell them; the market "
        "refuses your bids for them; gear and potions: you only buy them, for your trips):\n"
        if role is Role.ADVENTURER
        else "Market, for the goods you follow:\n"
    )
    market = follow + _json([view.to_payload() for view in observation.market])
    if observation.signals:
        market = (
            "Market signals (from the book and your own orders):\n"
            + "\n".join(observation.signals)
            + "\n\n"
            + market
        )

    directory = render_directory(observation.directory)

    sections = {
        SECTION_STATE: state,
        SECTION_MARKET: market,
        SECTION_DIRECTORY: directory,
        SECTION_INSTRUCTIONS: action_schema(condition),
    }

    if task is not None:
        sections[SECTION_TASK] = (
            "The Director has assigned you this task. Carry it out within your role: "
            "a task ends in the action that serves it; wait only if no action is "
            "worth taking now.\n" + _task_purpose(task) + _json(task)
        )

    # New messages and still-open proposals are both peer content, charged to
    # the p2p_content bucket. They never overlap: a proposal still in the inbox
    # is listed only as a new message (observation._open_proposals).
    p2p = []
    if observation.messages:
        p2p.append(
            "Messages addressed to you (a REQUEST or OFFER marked open can be "
            "answered until ticks_left reaches 0; one marked open false has "
            "expired, been replaced by a newer one or been answered, and is shown "
            "for information only):\n" + _json(observation.message_payloads())
        )
    if observation.open_proposals:
        p2p.append(
            OPEN_PROPOSALS_HEADER + " (received earlier, not yet answered; you can "
            "answer each until ticks_left reaches 0):\n" + _json(observation.open_proposals)
        )
    if p2p:
        sections[SECTION_P2P] = "\n\n".join(p2p)

    if observation.bulletin is not None:
        from rpg_economy.orchestration.global_view import render_global_summary

        sections[SECTION_BULLETIN] = "Global bulletin (published to everyone):\n" + (
            render_global_summary(observation.bulletin)
        )

    return Prompt(system=system, sections=sections)


def _task_purpose(task: dict) -> str:
    """The one-line purpose of the TASK's type (design v1.0 s3.1.1), if known."""
    from rpg_economy.domain import TaskType
    from rpg_economy.orchestration.protocol import TASK_PURPOSES

    try:
        task_type = TaskType(task.get("type"))
    except ValueError:
        return ""
    return f"{task_type.value}: {TASK_PURPOSES[task_type]}\n"


def _task_purposes_text() -> str:
    from rpg_economy.orchestration.protocol import TASK_PURPOSES

    return _json({task_type.value: text for task_type, text in TASK_PURPOSES.items()})


def agent_prompt_sources() -> tuple:
    """Every function whose source shapes an agent prompt's text.

    The code that assembles a prompt, not only its fixed strings: trait
    wording, section headers, the world rules' format and the task preamble
    live in these functions, as do the payload builders whose output is
    serialised into the prompt -- the TASK and its context references (A),
    peer messages (B, C), the market window, and the observation's own rows
    (open orders, own trades). The state section, tax_owed included, is
    rendered in build_agent_prompt itself.
    """
    from rpg_economy.agents.observation import (
        MarketView,
        Observation,
        _open_proposals,
        build_observation,
    )
    from rpg_economy.engine.market import Market
    from rpg_economy.messaging import Message
    from rpg_economy.orchestration.protocol import Result, ResultReference, Task

    return (
        build_agent_prompt,
        describe_traits,
        render_world,
        render_directory,
        _task_purpose,
        Task.to_payload,
        Task.to_agent_payload,
        ResultReference.public_view,
        Result.to_reference,
        Message.to_payload,
        Observation.message_payloads,
        MarketView.to_payload,
        _open_proposals,
        build_observation,
        Market.trades_of,
    )


def prompt_hashes() -> dict[str, str]:
    """Version hashes of every template (design v1.0 s15).

    Recorded in run metadata so a run can be tied to the exact wording that
    produced it: results can be sensitive to the wording.

    Computed once per process, as ``logging_.code_digest`` is: the source
    ``inspect.getsource`` reads is the file on disk, which may have been edited
    since it was imported. Tests that alter a template call
    ``prompt_hashes.cache_clear()``, which clears the Director's cache too.
    """
    return dict(_prompt_hashes())


@functools.cache
def _prompt_hashes() -> tuple[tuple[str, str], ...]:
    import inspect

    templates = {
        "agent_prompt_builder": "".join(inspect.getsource(f) for f in agent_prompt_sources()),
        "version": PROMPT_VERSION,
        "rules": RULES,
        "task_purposes": _task_purposes_text(),
        "action_schema": ACTION_SCHEMA,
        "action_schema_a": ACTION_SCHEMA_A,
        "open_proposals_header": OPEN_PROPOSALS_HEADER,
        "directory_format": render_directory(
            [{"agent_id": "a", "role": "r", "specialization": "s"}]
        ),
        **{f"role_brief.{role.value}": text for role, text in ROLE_BRIEF.items()},
    }
    return tuple(
        (name, hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
        for name, text in templates.items()
    )


# Clearing either module's cache clears both (logging_.metadata.register_hash_cache).
register_hash_cache(_prompt_hashes.cache_clear)
prompt_hashes.cache_clear = clear_hash_caches  # type: ignore[attr-defined]
