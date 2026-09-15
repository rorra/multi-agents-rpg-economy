"""Parsing and validating an agent's structured output (design v1.0 s7).

The model returns text; the engine needs a validated action. Everything that can
go wrong at that boundary is handled here and reported as one exception type, so
the retry policy has a single thing to catch.

Validation is deliberately strict. An action that "almost" parses is an invalid
action, not a salvage job: the design fixes a one-retry-then-WAIT policy, and
quietly repairing malformed output would hide exactly the coordination failures
the experiment is meant to measure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rpg_economy.agents.action import AgentAction
from rpg_economy.domain import (
    ORDER_ACTIONS,
    ActionType,
    Condition,
    MessageType,
    Priority,
    parse_good,
)

#: A JSON object embedded in surrounding prose, which models often emit.
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

#: Runs of whitespace or hyphens inside an id-like value.
_ID_SEPARATORS = re.compile(r"[\s\-]+")


def normalize_id(value: str) -> str:
    """Fold an id-like value (good, product, zone) to its canonical spelling.

    Lower-cased, with spaces and hyphens as underscores: "Arcane Crystal" and
    "arcane-crystal" name arcane_crystal. Every id in this economy is written
    that way, so the fold can only turn a formatting habit into the id it
    meant; a value that names nothing still fails validation. The Director's
    TASK params go through the same fold, so neither side is charged for it.
    """
    return _ID_SEPARATORS.sub("_", value.strip()).lower()


#: Actions that require each field.
_NEEDS_TARGET = frozenset(
    {
        ActionType.BUY,
        ActionType.SELL,
        ActionType.PRODUCE,
        ActionType.START_EXPEDITION,
        ActionType.EQUIP,
        ActionType.SEND_MESSAGE,
    }
)
_NEEDS_QUANTITY = frozenset({ActionType.BUY, ActionType.SELL, ActionType.PRODUCE})
_NEEDS_PRICE = frozenset({ActionType.BUY, ActionType.SELL})

#: What an agent may send peer to peer. TASK and RESULT belong to the Director
#: protocol of Condition A and are never agent traffic (design s6).
PEER_MESSAGE_TYPES = frozenset(
    {
        MessageType.REQUEST,
        MessageType.OFFER,
        MessageType.INFORM,
        MessageType.REJECT,
        MessageType.ACK,
    }
)
#: Message types that state terms, and so need a quantity and a price.
_MESSAGE_TERMS = frozenset({MessageType.REQUEST, MessageType.OFFER, MessageType.ACK})

#: Far above any price this economy produces. Only a guard: it keeps absurd
#: numbers (1e26) out of the engine's Decimal arithmetic, where they would stop
#: the run instead of being rejected as the invalid output they are.
MAX_PRICE = Decimal("1000000")

#: How much of short_reason is kept. It never affects the engine.
SHORT_REASON_LIMIT = 400


class InvalidAgentOutput(Exception):
    """Raised when a completion cannot be read as a valid action."""


class RawAction(BaseModel):
    """The wire form of design s7's action schema.

    Unknown keys are ignored and a null ``priority`` or ``short_reason`` takes its
    default. Neither carries a decision: tolerating them is not repairing a
    malformed action, and rejecting them would count a model's formatting habit
    as a coordination failure. Everything that does decide the action stays
    strict.
    """

    model_config = ConfigDict(extra="ignore")

    action: str
    target: str | None = None
    quantity: int | None = None
    price: Decimal | None = Field(default=None, allow_inf_nan=False, le=MAX_PRICE)
    recipient: str | None = None
    priority: str | None = Priority.MEDIUM.value
    short_reason: str | None = ""
    message_type: str | None = None
    #: cancel_order: the order to cancel. Also accepted in quantity, or as a
    #: number in target, which is where models also put it.
    order_id: int | None = None
    #: place_orders only: each item has the shape of a buy, sell or cancel_order.
    orders: list[Any] | None = None

    @field_validator("target", mode="before")
    @classmethod
    def _target_as_text(cls, value: Any) -> Any:
        # A bare number (an order id) is still a target named unambiguously.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    # RESULT fields, Condition A only (design v1.0 s3.1.2). Optional, and empty
    # is valid. `status` and `outcome` are engine-derived and deliberately not
    # here: an agent does not declare its own success. Held unchecked: they are
    # validated in Condition A alone (_result_fields), so B and C ignore them
    # as they ignore any other key outside their schema.
    needs: Any = None
    offers: Any = None
    observations: Any = None


def extract_json(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a completion."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", stripped).strip()

    candidates = [stripped]
    # The object that opens at the first brace: prose after it, even with
    # braces of its own, does not spoil it. Only the first: in a reply cut off
    # mid-object, a later brace opens an inner object (one order of a batch),
    # and taking that would execute a decision the model never finished.
    start = stripped.find("{")
    if start >= 0:
        try:
            payload, _ = json.JSONDecoder().raw_decode(stripped, start)
        except json.JSONDecodeError, ValueError:
            payload = None
        if isinstance(payload, dict):
            return payload
    match = _OBJECT_RE.search(stripped)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError, ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    raise InvalidAgentOutput(f"no JSON object found in completion: {text[:160]!r}")


@dataclass(frozen=True)
class ResultFields:
    """The agent-supplied part of a RESULT (design v1.0 s3.1.2), validated."""

    needs: list[Any]
    offers: list[Any]
    observations: list[Any]


#: The default cap on orders in one place_orders; the run uses its config value.
DEFAULT_MAX_ORDERS = 4


def parse_activation(
    text: str, *, condition: Condition, max_orders: int = DEFAULT_MAX_ORDERS
) -> tuple[AgentAction, ResultFields]:
    """Parse a completion into an action and its agent-supplied RESULT fields.

    In Condition A the RESULT fields are validated here, with the action: a
    malformed need or an observation outside the closed signal set is invalid
    output and takes the one retry (s3.1.2), rather than being dropped silently.
    Elsewhere they are not part of the protocol and are ignored.
    """
    raw = _parse_raw(text)
    action = _to_action(raw, condition, max_orders)
    if condition is not Condition.A:
        return action, ResultFields(needs=[], offers=[], observations=[])
    return action, _result_fields(raw)


#: The most entries each RESULT field may carry.
MAX_RESULT_ENTRIES = 10

NEED_FIELDS = {"good", "quantity", "priority"}
OFFER_FIELDS = {"good", "quantity", "price"}
OBSERVATION_FIELDS = {"signal", "good"}


def normalize_entry(item: Any, fields: set[str]) -> Any:
    """Tolerate what carries no decision, as the action parser does.

    Unknown keys are dropped and enum-like strings are case-folded; anything
    that does decide (a good that does not exist, a quantity of zero) is left
    for validation to refuse. The same leniency the agent's action gets, so no
    formatting habit costs Condition A alone.
    """
    if not isinstance(item, dict):
        return item
    out = {k: v for k, v in item.items() if k in fields}
    for key in ("good", "signal"):
        if isinstance(out.get(key), str):
            out[key] = normalize_id(out[key])
    if isinstance(out.get("priority"), str):
        out["priority"] = out["priority"].strip().lower()
    if out.get("priority") is None:
        out.pop("priority", None)
    # An offer priced at 0 names no price: read as null, which the schema
    # permits, rather than refused as a non-positive one.
    if "price" in out and _is_zero(out["price"]):
        out["price"] = None
    return out


def _is_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return Decimal(str(value)) == 0
    except InvalidOperation, ValueError:
        return False


def _result_fields(raw: RawAction) -> ResultFields:
    from rpg_economy.orchestration.protocol import Need, Offer, ResultObservation

    # The RESULT fields are a report beside the decision, not the decision: a
    # malformed entry is dropped on its own instead of discarding the action
    # with it. A field that is not a list reads as empty, and only the first
    # MAX_RESULT_ENTRIES entries are kept.
    def kept(items: Any, fields: set[str], model: Any) -> list[Any]:
        if not isinstance(items, list):
            return []
        out = []
        for item in items[:MAX_RESULT_ENTRIES]:
            try:
                out.append(model.model_validate(normalize_entry(item, fields)))
            except ValidationError, InvalidOperation, TypeError, ValueError:
                continue
        return out

    return ResultFields(
        needs=kept(raw.needs, NEED_FIELDS, Need),
        offers=kept(raw.offers, OFFER_FIELDS, Offer),
        observations=kept(raw.observations, OBSERVATION_FIELDS, ResultObservation),
    )


def parse_action(
    text: str, *, condition: Condition, max_orders: int = DEFAULT_MAX_ORDERS
) -> AgentAction:
    """Parse and validate a completion into an ``AgentAction``."""
    return _to_action(_parse_raw(text), condition, max_orders)


def _parse_raw(text: str) -> RawAction:
    payload = extract_json(text)
    try:
        return RawAction.model_validate(payload)
    except (ValidationError, InvalidOperation) as exc:
        raise InvalidAgentOutput(f"action failed schema validation: {exc}") from exc


def _to_action(
    raw: RawAction, condition: Condition, max_orders: int = DEFAULT_MAX_ORDERS
) -> AgentAction:
    # Goods, products and zones are lower-case ids: "Iron" names iron and
    # "arcane crystal" arcane_crystal, as they do in the Director's TASK
    # params. Orders inside a place_orders pass through here too, so they are
    # folded the same way.
    if isinstance(raw.target, str):
        raw = raw.model_copy(update={"target": normalize_id(raw.target)})

    # "start-expedition" and "Start Expedition" name start_expedition, as
    # "evaluate-expedition" names a TASK type for the Director.
    try:
        action_type = ActionType(raw.action.strip().lower().replace(" ", "_").replace("-", "_"))
    except ValueError:
        raise InvalidAgentOutput(f"unknown action {raw.action!r}") from None

    try:
        priority = Priority((raw.priority or Priority.MEDIUM.value).strip().lower())
    except ValueError:
        raise InvalidAgentOutput(f"unknown priority {raw.priority!r}") from None

    if action_type in _NEEDS_TARGET and not raw.target:
        raise InvalidAgentOutput(f"{action_type.value} requires a target")
    if action_type in _NEEDS_QUANTITY and (raw.quantity is None or raw.quantity <= 0):
        raise InvalidAgentOutput(f"{action_type.value} requires a positive quantity")
    if action_type in _NEEDS_PRICE and (raw.price is None or raw.price <= 0):
        raise InvalidAgentOutput(f"{action_type.value} requires a positive price")
    message_type = _message_type(raw, action_type)
    quantity = raw.quantity
    if action_type is ActionType.CANCEL_ORDER and raw.order_id is not None:
        quantity = raw.order_id
    if action_type is ActionType.CANCEL_ORDER and quantity is None:
        # The order id may also come in target: it still names the order
        # unambiguously.
        if raw.target is not None and str(raw.target).strip().lstrip("#").isdigit():
            quantity = int(str(raw.target).strip().lstrip("#"))
        else:
            raise InvalidAgentOutput("cancel_order requires the order id in order_id")

    # A blank or "none" recipient names nobody, in every condition.
    recipient = raw.recipient
    if isinstance(recipient, str) and recipient.strip().lower() in {"", "none", "null"}:
        recipient = None

    # In Condition A economic agents never emit peer-to-peer messages (design
    # v1.0 s3.1). A message action there is invalid output and takes the
    # ordinary retry-then-WAIT path. Any other action ignores the recipient,
    # as it does in B and C: a stray one carries no decision.
    if condition is Condition.A:
        if action_type is ActionType.SEND_MESSAGE:
            raise InvalidAgentOutput(
                "agents do not send peer-to-peer messages in Condition A; report the "
                "need through the RESULT instead"
            )
        recipient = None
    elif action_type is ActionType.SEND_MESSAGE:
        _check_message(raw.model_copy(update={"recipient": recipient}), message_type)

    orders = _orders(raw, action_type, condition, max_orders)

    return AgentAction(
        action=action_type,
        target=raw.target,
        quantity=quantity,
        price=raw.price,
        recipient=recipient,
        priority=priority,
        # Recorded for analysis only, so an overlong one is cut, not refused.
        short_reason=(raw.short_reason or "")[:SHORT_REASON_LIMIT],
        message_type=message_type,
        orders=orders,
    )


def _orders(
    raw: RawAction, action_type: ActionType, condition: Condition, max_orders: int
) -> tuple[AgentAction, ...]:
    """The orders of a place_orders, each validated as the action it is."""
    if action_type is not ActionType.PLACE_ORDERS:
        # A stray orders list beside another action is ignored, like a stray
        # message_type: the action named is the one the agent chose.
        return ()
    if not raw.orders:
        raise InvalidAgentOutput("place_orders needs at least one order in orders")
    if len(raw.orders) > max_orders:
        raise InvalidAgentOutput(
            f"place_orders carries at most {max_orders} orders, got {len(raw.orders)}"
        )
    orders = []
    for item in raw.orders:
        if not isinstance(item, dict):
            raise InvalidAgentOutput(f"each order must be an object, got {item!r}")
        try:
            sub = RawAction.model_validate(
                {**item, "priority": raw.priority, "short_reason": "", "orders": None}
            )
        except (ValidationError, InvalidOperation) as exc:
            raise InvalidAgentOutput(f"order failed schema validation: {exc}") from exc
        order = _to_action(sub, condition, max_orders)
        if order.action not in ORDER_ACTIONS:
            raise InvalidAgentOutput(
                f"place_orders takes only buy, sell and cancel_order, got {order.action.value}"
            )
        orders.append(order)
    return tuple(orders)


def _message_type(raw: RawAction, action_type: ActionType) -> MessageType | None:
    if action_type is not ActionType.SEND_MESSAGE:
        # A blank, "none" or "null" message_type names no message, as a
        # recipient spelled that way names nobody.
        # A message_type on any other action names no message, whatever it
        # says: the action is not a message, so it carries no decision.
        return None
    try:
        kind = MessageType((raw.message_type or "").strip().upper())
    except ValueError:
        raise InvalidAgentOutput(f"unknown message_type {raw.message_type!r}") from None
    if kind not in PEER_MESSAGE_TYPES:
        raise InvalidAgentOutput(f"{kind.value} is not a peer-to-peer message type")
    return kind


def _check_message(raw: RawAction, kind: MessageType | None) -> None:
    """A peer message is structured: who, what kind, which good, on what terms."""
    if not raw.recipient:
        raise InvalidAgentOutput("send_message requires a recipient")
    try:
        parse_good(raw.target or "")
    except ValueError:
        raise InvalidAgentOutput(
            f"send_message needs the good it is about in target, got {raw.target!r}"
        ) from None
    if kind in _MESSAGE_TERMS:
        if raw.quantity is None or raw.quantity <= 0:
            raise InvalidAgentOutput(f"a {kind.value} needs a positive quantity")
        if raw.price is None or raw.price <= 0:
            raise InvalidAgentOutput(f"a {kind.value} needs a positive price")
