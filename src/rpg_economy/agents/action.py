"""The agent action schema (design v1.0 s7).

``{action, target, quantity, price, recipient, priority, short_reason}`` is the
structured output every activated agent produces, in every condition. Rule-based
heuristics and LLM agents fill the same shape, so the executor below is written
once and reused.

``short_reason`` is stored for qualitative analysis only. It never affects engine
behaviour, and it is stripped from anything serialized into the Director's
context (design v1.0 s3.1.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from rpg_economy.domain import MONEY_QUANTUM, ActionType, MessageType, Priority

#: Money is quantized to this before reaching the market, so prices agents
#: compute never carry a tail of insignificant digits into the trade log.
PRICE_QUANTUM = MONEY_QUANTUM


@dataclass(frozen=True)
class AgentAction:
    """One activation's structured decision."""

    action: ActionType
    target: str | None = None
    quantity: int | None = None
    price: Decimal | None = None
    recipient: str | None = None
    priority: Priority = Priority.MEDIUM
    short_reason: str = ""
    #: send_message only: REQUEST, OFFER, INFORM, REJECT or ACK. ``target`` then
    #: names the good the message is about (design s6).
    message_type: MessageType | None = None
    #: place_orders only: the buy, sell and cancel_order orders placed together,
    #: each an action of its own validated and executed on its own.
    orders: tuple[AgentAction, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Log form. Includes ``short_reason``; Director-facing views must not."""
        return {
            "action": self.action.value,
            "target": self.target,
            "quantity": self.quantity,
            "price": None if self.price is None else str(self.price),
            "recipient": self.recipient,
            "priority": self.priority.value,
            "short_reason": self.short_reason,
            "message_type": None if self.message_type is None else self.message_type.value,
            "orders": [order.to_order_payload() for order in self.orders] or None,
        }

    def to_order_payload(self) -> dict[str, object]:
        """One order of a place_orders: what it does, without the envelope."""
        return {
            "action": self.action.value,
            "target": self.target,
            "quantity": self.quantity,
            "price": None if self.price is None else str(self.price),
        }

    def to_director_payload(self) -> dict[str, object]:
        """Director-facing form: the structured action without free text.

        Design s3.1.3 forbids the Director from reaching agents' private
        reasoning, and s3.1.2 forbids free text that could act as a covert
        instruction.
        """
        payload = self.to_payload()
        del payload["short_reason"]
        return payload


def wait(reason: str = "") -> AgentAction:
    return AgentAction(action=ActionType.WAIT, short_reason=reason)
