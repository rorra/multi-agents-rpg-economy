"""Applies a structured agent action to the engine.

The separation the design insists on (s2) runs through here: the agent decides
*what it attempts*; the engine validates inventory, gold, recipes, capacity and
randomness, and decides what actually happens. An action that fails validation is
recorded and discarded -- it never half-executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from rpg_economy.agents.action import PRICE_QUANTUM, AgentAction
from rpg_economy.domain import (
    ActionType,
    Condition,
    MessageType,
    OrderSide,
    Product,
    RejectionCode,
    parse_good,
)
from rpg_economy.engine import (
    Engine,
    ExpeditionRejected,
    OrderRejected,
    ProductionRejected,
)
from rpg_economy.logging_ import EventType
from rpg_economy.messaging import MessageBus, MessageRejected, NotAReply

#: What an engine refusal can raise. Decimal overflow and friends are among
#: them: an impossible number is a rejected action, never the end of the run.
#: ``EngineInvariantError`` is deliberately absent: a broken state invariant
#: may surface after goods or gold already moved, so it must end the run, not
#: be logged as a refused action.
_REFUSALS = (
    OrderRejected,
    ProductionRejected,
    ExpeditionRejected,
    MessageRejected,
    ValueError,
    KeyError,
    ArithmeticError,
)


@dataclass(frozen=True)
class ExecutionOutcome:
    """What the engine made of an attempted action."""

    accepted: bool
    #: True when the action changed canonical state or placed an order. Feeds the
    #: wasted-activation predicate of design v1.0 s11.2.
    engine_operation: bool
    #: True when the action emitted a peer-to-peer message. Also feeds that predicate:
    #: an activation that only communicated is not wasted.
    message_emitted: bool = False
    #: The engine's own words. Logged, and shown to the acting agent only.
    detail: str = ""
    #: The same refusal as a closed code, safe to report onwards (RESULT).
    code: RejectionCode | None = None
    #: place_orders only: per order, what it was and whether the engine took it.
    order_results: tuple[dict[str, object], ...] = ()

    @property
    def wasted(self) -> bool:
        """The wasted-activation predicate (design v1.0 s11.2).

        Wasted iff the final action was WAIT, or nothing was accepted by the
        engine and no message was sent. An order that later expires unfilled is
        NOT wasted: whether it fills depends on other agents, so charging it
        back here would make coordination efficiency a function of market
        outcomes rather than of coordination.
        """
        return not (self.engine_operation or self.message_emitted)


#: The only actions a place_orders may carry.
_ORDER_ACTIONS = frozenset({ActionType.BUY, ActionType.SELL, ActionType.CANCEL_ORDER})


def execute(
    engine: Engine,
    agent_id: str,
    action: AgentAction,
    *,
    messages: MessageBus | None = None,
    condition: Condition = Condition.B,
    activation: dict[str, object] | None = None,
) -> ExecutionOutcome:
    """Validate and apply ``action``, logging the attempt and its result.

    ``activation`` is the caller's key for the activation this action belongs
    to (``{tick, agent_id, seq}``). When given, it is added under
    ``"activation"`` to every event logged here, so they join to the rest of
    the activation's record.
    """
    engine.log_event(
        EventType.AGENT_OUTPUT,
        agent_id=agent_id,
        payload=_keyed(action.to_payload(), activation),
    )
    if action.action is ActionType.PLACE_ORDERS:
        return _execute_orders(engine, agent_id, action, messages, condition, activation)

    try:
        operated, messaged = _dispatch(engine, agent_id, action, messages, condition)
    except _REFUSALS as exc:
        code = rejection_code(exc)
        engine.log_event(
            EventType.ACTION_REJECTED,
            agent_id=agent_id,
            payload=_keyed(
                {"action": action.action.value, "reason": str(exc), "code": code.value},
                activation,
            ),
        )
        return ExecutionOutcome(accepted=False, engine_operation=False, detail=str(exc), code=code)

    engine.log_event(
        EventType.ACTION_EXECUTED,
        agent_id=agent_id,
        payload=_keyed({"action": action.action.value, "target": action.target}, activation),
    )
    return ExecutionOutcome(accepted=True, engine_operation=operated, message_emitted=messaged)


def _keyed(payload: dict[str, object], activation: dict[str, object] | None) -> dict[str, object]:
    """``payload`` with the activation key added, when there is one."""
    if activation is None:
        return payload
    return {**payload, "activation": dict(activation)}


def _malformed_batch(action: AgentAction, max_orders: int) -> str | None:
    """Why a place_orders cannot be executed at all, or None when it can.

    Parsing already enforces both rules; this is the engine-side check that
    does not trust it, so a caller that builds an action by hand cannot slip
    an expedition or an oversized batch through place_orders.
    """
    if len(action.orders) > max_orders:
        return f"place_orders carries {len(action.orders)} orders; at most {max_orders} are allowed"
    stray = sorted({o.action.value for o in action.orders if o.action not in _ORDER_ACTIONS})
    if stray:
        return f"place_orders may carry only buy, sell and cancel_order, not {stray}"
    return None


def _execute_orders(
    engine: Engine,
    agent_id: str,
    action: AgentAction,
    messages: MessageBus | None,
    condition: Condition,
    activation: dict[str, object] | None = None,
) -> ExecutionOutcome:
    """Execute each order of a place_orders on its own.

    One bad order does not void the others: those the engine takes go on the
    book, the rest are rejected with their code. The activation counts as
    accepted when at least one order went through. A batch that breaks the
    shape rules (only buy, sell and cancel_order; at most
    ``market.max_orders_per_activation`` orders) is rejected whole, before any
    order reaches the book.
    """
    problem = _malformed_batch(action, engine.config.market.max_orders_per_activation)
    if problem is not None:
        engine.log_event(
            EventType.ACTION_REJECTED,
            agent_id=agent_id,
            payload=_keyed(
                {
                    "action": action.action.value,
                    "reason": problem,
                    "code": RejectionCode.INVALID_ORDER.value,
                },
                activation,
            ),
        )
        return ExecutionOutcome(
            accepted=False,
            engine_operation=False,
            detail=problem,
            code=RejectionCode.INVALID_ORDER,
        )
    results: list[dict[str, object]] = []
    details: list[str] = []
    codes: list[RejectionCode] = []
    for index, order in enumerate(action.orders):
        try:
            _dispatch(engine, agent_id, order, messages, condition)
        except _REFUSALS as exc:
            code = rejection_code(exc)
            engine.log_event(
                EventType.ACTION_REJECTED,
                agent_id=agent_id,
                payload=_keyed(
                    {
                        "action": order.action.value,
                        "target": order.target,
                        "reason": str(exc),
                        "code": code.value,
                        "order_index": index,
                    },
                    activation,
                ),
            )
            details.append(f"order {index + 1}: {exc}")
            codes.append(code)
            results.append({**order.to_order_payload(), "accepted": False, "rejection": code.value})
            continue
        engine.log_event(
            EventType.ACTION_EXECUTED,
            agent_id=agent_id,
            payload=_keyed(
                {"action": order.action.value, "target": order.target, "order_index": index},
                activation,
            ),
        )
        results.append({**order.to_order_payload(), "accepted": True, "rejection": None})
    accepted = any(result["accepted"] for result in results)
    return ExecutionOutcome(
        accepted=accepted,
        engine_operation=accepted,
        detail="; ".join(details),
        code=None if accepted or not codes else codes[0],
        order_results=tuple(results),
    )


def _order_price(price: Decimal) -> Decimal:
    """The limit price quantized to the market's money unit.

    A price too large to quantize is a malformed order: it would otherwise
    surface as a bare InvalidOperation and be misread as a bad quantity.
    """
    try:
        return Decimal(price).quantize(PRICE_QUANTUM)
    except InvalidOperation:
        raise OrderRejected(f"order price {price} is too large to trade") from None


def rejection_code(exc: BaseException) -> RejectionCode:
    """Classify an engine refusal into the closed vocabulary of RejectionCode."""
    text = str(exc)
    if isinstance(exc, NotAReply):
        return RejectionCode.NOT_A_REPLY
    if isinstance(exc, MessageRejected):
        return RejectionCode.MESSAGE_REJECTED
    # Named explicitly: later substring rules must never claim it.
    if "own resting" in text:
        return RejectionCode.INVALID_ORDER
    # A non-positive or oversized limit price is a malformed order, not a bad quantity.
    if "price must be positive" in text or "too large to trade" in text:
        return RejectionCode.INVALID_ORDER
    if isinstance(exc, ArithmeticError) or "must be positive" in text:
        return RejectionCode.INVALID_QUANTITY
    if "cannot bid" in text:
        return RejectionCode.INSUFFICIENT_GOLD
    if "cannot sell" in text or "no unreserved" in text:
        return RejectionCode.INSUFFICIENT_INVENTORY
    if "needs" in text and "unreserved" in text:
        return RejectionCode.MISSING_INPUTS
    if "capacity" in text:
        return RejectionCode.NO_CAPACITY
    if (
        "cannot craft" in text
        or "not an artisan" in text
        or "not an adventurer" in text
        or "cannot buy raw materials" in text
        or "may not offer products" in text
    ):
        return RejectionCode.NOT_ALLOWED
    if "already away" in text:
        return RejectionCode.AGENT_AWAY
    if "is resting" in text:
        return RejectionCode.RESTING
    if text.startswith("order "):
        return RejectionCode.INVALID_ORDER
    if "requires" in text:
        return RejectionCode.MISSING_FIELDS
    if isinstance(exc, (KeyError, ValueError, ProductionRejected, ExpeditionRejected)):
        return RejectionCode.INVALID_TARGET
    return RejectionCode.OTHER


def _dispatch(
    engine: Engine,
    agent_id: str,
    action: AgentAction,
    messages: MessageBus | None,
    condition: Condition,
) -> tuple[bool, bool]:
    """Apply the action; return (engine operation, message emitted)."""
    match action.action:
        case ActionType.WAIT:
            return False, False

        case ActionType.BUY | ActionType.SELL:
            if action.target is None or action.quantity is None or action.price is None:
                raise ValueError(f"{action.action} requires target, quantity and price")
            side = OrderSide.BUY if action.action is ActionType.BUY else OrderSide.SELL
            engine.submit_order(
                agent_id,
                side,
                parse_good(action.target),
                action.quantity,
                _order_price(action.price),
            )
            return True, False

        case ActionType.CANCEL_ORDER:
            if action.quantity is None:
                raise ValueError("cancel_order requires the order id in `quantity`")
            # Only the agent's own order: ids are small sequential integers,
            # and a misremembered one must not cancel a peer's order.
            engine.market.cancel(action.quantity, agent_id=agent_id)
            return True, False

        case ActionType.PRODUCE:
            if action.target is None or action.quantity is None:
                raise ValueError("produce requires target and quantity")
            engine.produce(agent_id, Product(action.target), action.quantity)
            return True, False

        case ActionType.START_EXPEDITION:
            if action.target is None:
                raise ValueError("start_expedition requires a zone id in `target`")
            engine.start_expedition(agent_id, action.target)
            return True, False

        case ActionType.EQUIP:
            if action.target is None:
                raise ValueError("equip requires a product in `target`")
            engine.equip(agent_id, Product(action.target))
            return True, False

        case ActionType.SEND_MESSAGE:
            # Condition A has no peer-to-peer traffic at all (design v1.0 s3.1);
            # parsing rejects it earlier, and this is the belt to that braces.
            if condition is Condition.A:
                raise ValueError("agents do not send peer-to-peer messages in Condition A")
            if messages is None:
                raise ValueError("no message bus is available in this run")
            if action.recipient is None:
                raise ValueError("send_message requires a recipient")
            if action.message_type is None or action.target is None:
                raise ValueError("send_message requires a message type and a good")
            message_type = action.message_type
            placed = False
            if message_type is MessageType.ACK:
                # Accepting is committing: the acceptor's matching order goes on
                # the book in the same activation. An ACK alone left the deal to
                # a later activation that often did something else. The order is
                # placed first, so an ACK the agent cannot back is never sent.
                proposal = messages.open_proposal(agent_id, action.recipient, action.target)
                if proposal is None:
                    raise NotAReply(
                        f"ACK answers a REQUEST or OFFER received from {action.recipient} "
                        f"about {action.target}; there is none open"
                    )
                if action.quantity is None or action.price is None:
                    raise ValueError("an ACK requires the quantity and price of its order")
                side = OrderSide.BUY if proposal is MessageType.OFFER else OrderSide.SELL
                engine.submit_order(
                    agent_id,
                    side,
                    parse_good(action.target),
                    action.quantity,
                    _order_price(action.price),
                )
                placed = True
            sent = messages.send(
                sender_id=agent_id,
                recipient_id=action.recipient,
                message_type=message_type,
                tick=engine.tick,
                priority=action.priority,
                payload={
                    "good": action.target,
                    "quantity": action.quantity,
                    "price": None if action.price is None else str(action.price),
                },
            )
            engine.log_event(
                EventType.MESSAGE_SENT,
                agent_id=agent_id,
                payload={
                    # The bus id, so message_delivered can be joined to its send.
                    "message_id": sent.message_id,
                    "recipient_id": action.recipient,
                    "type": message_type.value,
                    "good": action.target,
                    "quantity": action.quantity,
                    "price": None if action.price is None else str(action.price),
                },
            )
            return placed, True

    raise ValueError(f"unhandled action {action.action}")
