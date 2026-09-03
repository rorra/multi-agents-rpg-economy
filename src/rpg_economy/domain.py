"""Canonical domain vocabulary.

These enums are the single source of truth for every domain term in the code.
They are the English side of the es <-> en mapping in ``docs/glossary.md``; the
Spanish side belongs to the design document and to rendered output only.

Stored data (event logs, metrics files) contains these language-neutral values.
Translation happens at render time, never at write time (IMPLEMENTATION.md section 2).
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

#: The money unit: every price, charge and freeze is rounded to it.
MONEY_QUANTUM = Decimal("0.01")


class Condition(StrEnum):
    """Experimental condition -- the independent variable (design v1.0 s3)."""

    A = "A"  # centralized, LLM Director
    B = "B"  # decentralized, peer-to-peer
    C = "C"  # decentralized + passive global bulletin


class ExperimentRegime(StrEnum):
    """Budget regime (design v1.0 s8.4)."""

    EQUAL_ACTIVITY = "experiment_1"  # same K activations per macro-cycle
    EQUAL_COST = "experiment_2"  # same total token budget per run


class Scenario(StrEnum):
    """Shock scenarios (design v1.0 s10). ``noisy_info`` is optional and is only
    admissible if defined before the final runs."""

    RESOURCE_SCARCITY = "resource_scarcity"  # primary
    DEMAND_SURGE = "demand_surge"
    PRODUCTIVITY_DROP = "productivity_drop"
    LIQUIDITY_LOSS = "liquidity_loss"
    NOISY_INFO = "noisy_info"


class Resource(StrEnum):
    """Raw materials (design v1.0 s5.1)."""

    IRON = "iron"
    WOOD = "wood"
    LEATHER = "leather"
    HERBS = "herbs"
    ARCANE_CRYSTAL = "arcane_crystal"


class Product(StrEnum):
    """Crafted products (design v1.0 s5.1)."""

    MELEE_WEAPON = "melee_weapon"
    ARMOR = "armor"
    BOW_GEAR = "bow_gear"
    STAFF_FOCUS = "staff_focus"
    HEALTH_POTION = "health_potion"
    MANA_POTION = "mana_potion"


class Role(StrEnum):
    """Economic roles (design v1.0 s4)."""

    ADVENTURER = "adventurer"
    ARTISAN = "artisan"


class AdventurerSubclass(StrEnum):
    WARRIOR = "warrior"
    MAGE = "mage"
    RANGER = "ranger"


class ArtisanSpecialization(StrEnum):
    SMITHING = "smithing"
    WOODWORKING = "woodworking"
    ALCHEMY = "alchemy"


class TaskType(StrEnum):
    """Closed TASK vocabulary, Condition A only (design v1.0 s3.1.1)."""

    ANALYZE_MARKET = "ANALYZE_MARKET"
    EVALUATE_EXPEDITION = "EVALUATE_EXPEDITION"
    EVALUATE_PRODUCTION = "EVALUATE_PRODUCTION"
    MANAGE_PROCUREMENT = "MANAGE_PROCUREMENT"
    MANAGE_INVENTORY = "MANAGE_INVENTORY"
    ADDRESS_IMBALANCE = "ADDRESS_IMBALANCE"


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RejectionCode(StrEnum):
    """Why the engine refused an action, as a closed code (design v1.0 s3.1.2).

    What a RESULT reports instead of the engine's message. The message names
    agents and quotes their gold and stock; carried to the Director and on into
    other agents' TASK context, it would show them holdings the design keeps
    private (s5.2).
    """

    INSUFFICIENT_GOLD = "insufficient_gold"
    INSUFFICIENT_INVENTORY = "insufficient_inventory"
    MISSING_INPUTS = "missing_inputs"
    NO_CAPACITY = "no_capacity"
    NOT_ALLOWED = "not_allowed"
    #: On an expedition: cannot equip or set out (it can still trade).
    AGENT_AWAY = "agent_away"
    #: Back in town but still recovering: can trade and equip, cannot set out yet.
    RESTING = "resting"
    INVALID_ORDER = "invalid_order"
    INVALID_TARGET = "invalid_target"
    INVALID_QUANTITY = "invalid_quantity"
    MISSING_FIELDS = "missing_fields"
    MESSAGE_REJECTED = "message_rejected"
    NOT_A_REPLY = "not_a_reply"
    OTHER = "other"


class ObservationSignal(StrEnum):
    """What a RESULT observation may report about a good (design v1.0 s3.1.2).

    A closed set, so an observation is a structured signal rather than free
    text that could carry instructions or reasoning to the Director.
    """

    PRICE_RISING = "price_rising"
    PRICE_FALLING = "price_falling"
    SCARCE = "scarce"
    SURPLUS = "surplus"
    NO_BUYERS = "no_buyers"
    NO_SELLERS = "no_sellers"


class ResultStatus(StrEnum):
    """RESULT status values (design v1.0 s3.1.2)."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class MessageType(StrEnum):
    """The only admissible message types (design v1.0 s6). No free chat.

    TASK and RESULT exist only in Condition A. Economic agents never emit
    peer-to-peer messages in Condition A (design v1.0 s3.1).
    """

    REQUEST = "REQUEST"
    OFFER = "OFFER"
    INFORM = "INFORM"
    TASK = "TASK"
    RESULT = "RESULT"
    REJECT = "REJECT"
    ACK = "ACK"


class Trigger(StrEnum):
    """Mechanical activation triggers, Conditions B and C only (design v1.0 s8.2).

    Evaluated by the engine with no LLM involvement whatsoever. The Condition C
    bulletin fires none of these.
    """

    T1_MESSAGE_RECEIVED = "T1_message_received"
    T2_ORDER_EVENT = "T2_order_event"
    T3_PRICE_MOVE = "T3_price_move"
    T4_OWN_STATE_THRESHOLD = "T4_own_state_threshold"
    T5_INACTIVITY = "T5_inactivity"


class RiskLevel(StrEnum):
    """Expedition risk levels (design v1.0 s5.3)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TokenBucket(StrEnum):
    """Token accounting buckets (design v1.0 s8.4, s15).

    ``P2P_CONTENT`` is structurally absent in Condition A and ``BULLETIN`` in
    Conditions A and B (design v1.0 s3.1, s3.3).
    """

    AGENT = "agent"
    DIRECTOR = "director"
    P2P_CONTENT = "p2p_content"
    BULLETIN = "bulletin"


class OrderSide(StrEnum):
    """Side of a limit order in the central order book (design v1.0 s5.2)."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    """Lifecycle of an order. Every transition is logged (design v1.0 s5.2)."""

    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ActionType(StrEnum):
    """What an activated agent can attempt.

    The design lists the decisions each role makes (s4.1-4.2) without naming an
    action vocabulary; this is that list made executable. The engine validates
    every attempt against the canonical state -- an action here is a *proposal*,
    never an effect.
    """

    WAIT = "wait"
    BUY = "buy"
    SELL = "sell"
    CANCEL_ORDER = "cancel_order"
    PRODUCE = "produce"
    START_EXPEDITION = "start_expedition"
    EQUIP = "equip"
    SEND_MESSAGE = "send_message"
    #: Several market orders (buy, sell, cancel_order) in one activation.
    PLACE_ORDERS = "place_orders"


#: The actions that are market orders, and so may go together in place_orders.
ORDER_ACTIONS = frozenset({ActionType.BUY, ActionType.SELL, ActionType.CANCEL_ORDER})


class ExpeditionOutcome(StrEnum):
    """How an expedition resolved (design v1.0 s5.3).

    Failure costs time and yields little; it is never permanent death, which
    s14 places out of scope.
    """

    SUCCESS = "success"
    FAILURE = "failure"


#: Anything tradable: raw materials and crafted products alike.
type Good = Resource | Product

#: Every tradable good, in a fixed order.
ALL_GOODS: tuple[Good, ...] = (*Resource, *Product)

_GOODS_BY_VALUE: dict[str, Good] = {good.value: good for good in ALL_GOODS}


def parse_good(value: str) -> Good:
    """Resolve a good's stored string value back to its enum member."""
    try:
        return _GOODS_BY_VALUE[value]
    except KeyError:
        raise ValueError(f"unknown good {value!r}") from None


#: Fixed resource -> main finished product mapping (design v1.0 s11.3).
#: Used by the propagation analysis and by the productivity-drop scenario.
#: May be adjusted once at pilot close; immutable after final runs begin.
RESOURCE_TO_MAIN_PRODUCT: dict[Resource, Product] = {
    Resource.IRON: Product.MELEE_WEAPON,
    Resource.WOOD: Product.BOW_GEAR,
    Resource.LEATHER: Product.ARMOR,
    Resource.HERBS: Product.HEALTH_POTION,
    Resource.ARCANE_CRYSTAL: Product.STAFF_FOCUS,
}
