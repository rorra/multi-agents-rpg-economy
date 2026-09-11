"""Structured messages and the static public directory (design v1.0 s6, s6.1)."""

from __future__ import annotations

import pytest

from rpg_economy.domain import MessageType, Priority, Role
from rpg_economy.messaging import (
    P2P_MESSAGE_TYPES,
    Directory,
    DirectoryEntry,
    MessageBus,
    MessageRejected,
    NotAReply,
)


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus(
        Directory(
            [
                DirectoryEntry("warrior_1", Role.ADVENTURER, "warrior"),
                DirectoryEntry("smithing_1", Role.ARTISAN, "smithing"),
                DirectoryEntry("woodworking_1", Role.ARTISAN, "woodworking"),
            ]
        )
    )


def test_a_message_reaches_exactly_one_recipient(bus: MessageBus) -> None:
    bus.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.OFFER,
        tick=1,
    )
    assert bus.unread_count("warrior_1") == 1
    assert bus.unread_count("smithing_1") == 0
    assert bus.unread_count("woodworking_1") == 0


def test_only_the_closed_set_of_types_is_accepted(bus: MessageBus) -> None:
    # ACK and REJECT each answer an open proposal, so one is opened before each.
    for message_type in P2P_MESSAGE_TYPES:
        if message_type in {MessageType.ACK, MessageType.REJECT}:
            bus.send(
                sender_id="warrior_1",
                recipient_id="woodworking_1",
                message_type=MessageType.OFFER,
                tick=1,
                payload={"good": "iron"},
            )
        bus.send(
            sender_id="woodworking_1",
            recipient_id="warrior_1",
            message_type=message_type,
            tick=1,
            payload={"good": "iron"},
        )
    assert bus.unread_count("warrior_1") == len(P2P_MESSAGE_TYPES)


def _offer(bus: MessageBus, sender: str, recipient: str, good: str = "iron") -> None:
    bus.send(
        sender_id=sender,
        recipient_id=recipient,
        message_type=MessageType.OFFER,
        tick=1,
        payload={"good": good, "quantity": 1, "price": "10"},
    )


@pytest.mark.parametrize("closing", [MessageType.ACK, MessageType.REJECT])
def test_an_answer_needs_a_proposal_to_answer(bus: MessageBus, closing: MessageType) -> None:
    with pytest.raises(NotAReply):
        bus.send(
            sender_id="woodworking_1",
            recipient_id="warrior_1",
            message_type=closing,
            tick=1,
            payload={"good": "iron"},
        )


@pytest.mark.parametrize("closing", [MessageType.ACK, MessageType.REJECT])
def test_a_proposal_is_answered_once(bus: MessageBus, closing: MessageType) -> None:
    _offer(bus, "warrior_1", "woodworking_1")
    answer = {"sender_id": "woodworking_1", "recipient_id": "warrior_1", "tick": 2}
    bus.send(message_type=closing, payload={"good": "iron"}, **answer)
    with pytest.raises(NotAReply):
        bus.send(message_type=MessageType.ACK, payload={"good": "iron"}, **answer)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (MessageType.ACK, MessageType.ACK),
        (MessageType.ACK, MessageType.REJECT),
        (MessageType.REJECT, MessageType.ACK),
        (MessageType.REJECT, MessageType.REJECT),
    ],
)
def test_an_answer_is_never_answered(
    bus: MessageBus, first: MessageType, second: MessageType
) -> None:
    """The ping-pong: each answer fired the other side's T1, forever."""
    _offer(bus, "warrior_1", "woodworking_1")
    bus.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=first,
        tick=2,
        payload={"good": "iron"},
    )
    with pytest.raises(NotAReply):
        bus.send(
            sender_id="warrior_1",
            recipient_id="woodworking_1",
            message_type=second,
            tick=3,
            payload={"good": "iron"},
        )


def test_an_answer_must_be_about_the_proposed_good(bus: MessageBus) -> None:
    _offer(bus, "warrior_1", "woodworking_1", good="iron")
    with pytest.raises(NotAReply):
        bus.send(
            sender_id="woodworking_1",
            recipient_id="warrior_1",
            message_type=MessageType.ACK,
            tick=2,
            payload={"good": "leather"},
        )


def test_only_the_recipient_of_a_proposal_can_answer_it(bus: MessageBus) -> None:
    _offer(bus, "warrior_1", "woodworking_1")
    with pytest.raises(NotAReply):
        bus.send(
            sender_id="smithing_1",
            recipient_id="warrior_1",
            message_type=MessageType.ACK,
            tick=2,
            payload={"good": "iron"},
        )


def test_an_inform_opens_nothing_to_answer(bus: MessageBus) -> None:
    bus.send(
        sender_id="warrior_1",
        recipient_id="woodworking_1",
        message_type=MessageType.INFORM,
        tick=1,
        payload={"good": "iron"},
    )
    with pytest.raises(NotAReply):
        bus.send(
            sender_id="woodworking_1",
            recipient_id="warrior_1",
            message_type=MessageType.ACK,
            tick=2,
            payload={"good": "iron"},
        )


@pytest.mark.parametrize("message_type", [MessageType.TASK, MessageType.RESULT])
def test_director_traffic_is_not_peer_to_peer(bus: MessageBus, message_type: MessageType) -> None:
    """TASK and RESULT belong to Condition A's Director channel (design v1.0 s3.1)."""
    with pytest.raises(MessageRejected, match="not peer-to-peer"):
        bus.send(
            sender_id="woodworking_1",
            recipient_id="warrior_1",
            message_type=message_type,
            tick=1,
        )


def test_unknown_participants_are_rejected(bus: MessageBus) -> None:
    with pytest.raises(MessageRejected, match="unknown recipient"):
        bus.send(
            sender_id="woodworking_1",
            recipient_id="nobody",
            message_type=MessageType.INFORM,
            tick=1,
        )
    with pytest.raises(MessageRejected, match="unknown sender"):
        bus.send(
            sender_id="nobody",
            recipient_id="warrior_1",
            message_type=MessageType.INFORM,
            tick=1,
        )


def test_an_agent_cannot_message_itself(bus: MessageBus) -> None:
    with pytest.raises(MessageRejected, match="cannot message itself"):
        bus.send(
            sender_id="warrior_1",
            recipient_id="warrior_1",
            message_type=MessageType.INFORM,
            tick=1,
        )


def test_draining_an_inbox_empties_it(bus: MessageBus) -> None:
    bus.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.REQUEST,
        tick=1,
    )
    assert len(bus.drain("warrior_1")) == 1
    assert bus.unread_count("warrior_1") == 0


def test_inbox_preserves_arrival_order(bus: MessageBus) -> None:
    for tick in range(3):
        bus.send(
            sender_id="woodworking_1",
            recipient_id="warrior_1",
            message_type=MessageType.INFORM,
            tick=tick,
            payload={"n": tick},
        )
    assert [m.payload["n"] for m in bus.inbox("warrior_1")] == [0, 1, 2]


def test_every_message_is_counted_as_an_interaction(bus: MessageBus) -> None:
    """Design v1.0 s6.1: each P2P message carries an interaction cost."""
    for _ in range(4):
        bus.send(
            sender_id="woodworking_1",
            recipient_id="warrior_1",
            message_type=MessageType.OFFER,
            tick=1,
        )
    assert len(bus.sent) == 4


def test_directory_exposes_only_id_role_and_specialization(bus: MessageBus) -> None:
    """No gold, no inventory, no plans (design v1.0 s6.1)."""
    entry = bus.directory.entry("smithing_1")
    assert entry.role is Role.ARTISAN
    assert entry.specialization == "smithing"
    assert set(vars(entry)) == {"agent_id", "role", "specialization"}


def test_directory_is_complete_and_static(bus: MessageBus) -> None:
    assert bus.directory.agent_ids() == ["smithing_1", "warrior_1", "woodworking_1"]
    assert bus.directory.with_role(Role.ARTISAN) == ["smithing_1", "woodworking_1"]
    assert not hasattr(bus.directory, "discover")


def test_message_payload_round_trips(bus: MessageBus) -> None:
    message = bus.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.REQUEST,
        tick=3,
        priority=Priority.HIGH,
        payload={"good": "iron", "quantity": 5},
    )
    payload = message.to_payload()
    assert payload["type"] == "REQUEST"
    assert payload["priority"] == "high"
    assert payload["payload"]["good"] == "iron"


# ---------------------------------------------------------------------------
# Open proposals expire as orders do
# ---------------------------------------------------------------------------
def expiring_bus(ticks: int = 6, now: list[int] | None = None) -> tuple[MessageBus, list[int]]:
    clock = now if now is not None else [0]
    bus = MessageBus(
        Directory(
            [
                DirectoryEntry("warrior_1", Role.ADVENTURER, "warrior"),
                DirectoryEntry("smithing_1", Role.ARTISAN, "smithing"),
            ]
        ),
        proposal_expiry_ticks=ticks,
        clock=lambda: clock[0],
    )
    return bus, clock


def offer_iron(bus: MessageBus, tick: int) -> None:
    bus.send(
        sender_id="warrior_1",
        recipient_id="smithing_1",
        message_type=MessageType.OFFER,
        tick=tick,
        payload={"good": "iron", "quantity": 1, "price": "10"},
    )


@pytest.mark.parametrize("kind", [MessageType.ACK, MessageType.REJECT])
def test_a_proposal_can_be_answered_until_it_expires(kind: MessageType) -> None:
    bus, _ = expiring_bus(ticks=6)
    offer_iron(bus, tick=0)
    answer = bus.send(
        sender_id="smithing_1",
        recipient_id="warrior_1",
        message_type=kind,
        tick=5,
        payload={"good": "iron"},
    )
    assert answer.type is kind
    assert answer.recipient_id == "warrior_1"
