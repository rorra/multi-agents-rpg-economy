"""Structured peer-to-peer messages (design v1.0 s6).

There is no free chat. Every message is one of the closed set of types, carries a
structured payload, and has a single recipient, so that coordination is fully
observable and countable.

The Condition C bulletin is *not* a message. It is engine-published information,
it fires no trigger, and it never passes through this module (design v1.0 s3.3).
Keeping the bulletin structurally outside the message bus is what makes the
"bulletin never triggers T1" rule impossible to violate by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rpg_economy.domain import MessageType, Priority, Role

#: Message types economic agents may exchange peer-to-peer. TASK and RESULT are
#: Director traffic in Condition A and never P2P (design v1.0 s3.1).
P2P_MESSAGE_TYPES: frozenset[MessageType] = frozenset(
    {
        MessageType.REQUEST,
        MessageType.OFFER,
        MessageType.INFORM,
        MessageType.REJECT,
        MessageType.ACK,
    }
)


@dataclass(frozen=True)
class Message:
    """One structured message with a single recipient."""

    message_id: int
    sender_id: str
    recipient_id: str
    type: MessageType
    sent_tick: int
    priority: Priority = Priority.MEDIUM
    payload: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "type": self.type.value,
            "sent_tick": self.sent_tick,
            "priority": self.priority.value,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class DirectoryEntry:
    """What every agent knows about every other agent (design v1.0 s6.1).

    Id, role and specialization -- nothing else. No gold, no inventory, no plans,
    and no dynamic discovery in this version.
    """

    agent_id: str
    role: Role
    specialization: str | None
