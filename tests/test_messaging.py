"""Structured messages and the static public directory (design v1.0 s6, s6.1)."""

from __future__ import annotations

import pytest

from rpg_economy.domain import MessageType, Priority, Role, parse_good
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


@pytest.mark.parametrize("kind", [MessageType.ACK, MessageType.REJECT])
def test_an_answer_to_an_expired_proposal_is_not_a_reply(kind: MessageType) -> None:
    """Expired after the order expiry: refused like an answer to nothing."""
    bus, _ = expiring_bus(ticks=6)
    offer_iron(bus, tick=0)
    with pytest.raises(NotAReply):
        bus.send(
            sender_id="smithing_1",
            recipient_id="warrior_1",
            message_type=kind,
            tick=6,
            payload={"good": "iron"},
        )


def test_open_proposal_reads_the_clock_and_skips_expired_ones() -> None:
    bus, clock = expiring_bus(ticks=6)
    offer_iron(bus, tick=0)
    offer_iron(bus, tick=4)
    clock[0] = 5
    assert bus.open_proposal("smithing_1", "warrior_1", "iron") is MessageType.OFFER
    clock[0] = 6
    # The second replaced the first, and is open until its own expiry.
    assert bus.open_proposal("smithing_1", "warrior_1", "iron") is MessageType.OFFER
    clock[0] = 10
    assert bus.open_proposal("smithing_1", "warrior_1", "iron") is None
    assert bus.open_proposal("smithing_1", "warrior_1", "iron", tick=9) is None


def test_without_an_expiry_a_proposal_stays_open(bus: MessageBus) -> None:
    bus.send(
        sender_id="warrior_1",
        recipient_id="smithing_1",
        message_type=MessageType.OFFER,
        tick=0,
        payload={"good": "iron"},
    )
    assert bus.open_proposal("smithing_1", "warrior_1", "iron", tick=10_000) is MessageType.OFFER


def test_the_decentralized_orchestrator_sets_the_order_expiry_on_the_bus(tmp_path) -> None:
    from rpg_economy.agents.action import AgentAction
    from rpg_economy.agents.executor import execute
    from rpg_economy.domain import ActionType, Condition, RejectionCode
    from rpg_economy.engine import Engine, build_world
    from rpg_economy.logging_ import EventLogWriter
    from rpg_economy.orchestration import DecentralizedOrchestrator, build_directory
    from tests.conftest import make_config, make_metadata

    config = make_config(run={"condition": "B"})
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        messages = MessageBus(build_directory(engine))
        DecentralizedOrchestrator(config, engine, lambda e, a: None, messages=messages)
        expiry = config.market.order_expiry_ticks
        assert messages.proposal_expiry_ticks == expiry

        engine.state.account("warrior_1").receive(parse_good("iron"), 2)
        offer = AgentAction(
            action=ActionType.SEND_MESSAGE,
            message_type=MessageType.OFFER,
            target="iron",
            quantity=1,
            price="10",
            recipient="smithing_1",
        )
        assert execute(
            engine, "warrior_1", offer, messages=messages, condition=Condition.B
        ).accepted

        engine.state.tick += expiry
        ack = AgentAction(
            action=ActionType.SEND_MESSAGE,
            message_type=MessageType.ACK,
            target="iron",
            quantity=1,
            price="10",
            recipient="warrior_1",
        )
        outcome = execute(engine, "smithing_1", ack, messages=messages, condition=Condition.B)
        assert not outcome.accepted
        assert outcome.code is RejectionCode.NOT_A_REPLY
        # The ACK places no order when there is nothing open to accept.
        assert not engine.market.open_orders_of("smithing_1")


# ---------------------------------------------------------------------------
# At most one open proposal per (sender, recipient, good)
# ---------------------------------------------------------------------------
def _propose(bus: MessageBus, kind: MessageType, tick: int = 1, good: str = "iron", **ids) -> None:
    bus.send(
        sender_id=ids.get("sender", "warrior_1"),
        recipient_id=ids.get("recipient", "smithing_1"),
        message_type=kind,
        tick=tick,
        payload={"good": good, "quantity": 1, "price": "10"},
    )


def test_an_offer_then_a_request_leaves_only_the_request_open(bus: MessageBus) -> None:
    _propose(bus, MessageType.OFFER)
    _propose(bus, MessageType.REQUEST)
    assert bus.open_proposal("smithing_1", "warrior_1", "iron") is MessageType.REQUEST
    # Both were delivered: replacing closes the older proposal, not the message.
    assert bus.unread_count("smithing_1") == 2

    bus.send(
        sender_id="smithing_1",
        recipient_id="warrior_1",
        message_type=MessageType.ACK,
        tick=2,
        payload={"good": "iron"},
    )
    assert bus.open_proposal("smithing_1", "warrior_1", "iron") is None
    # The replaced OFFER is not left behind to be answered.
    with pytest.raises(NotAReply):
        bus.send(
            sender_id="smithing_1",
            recipient_id="warrior_1",
            message_type=MessageType.REJECT,
            tick=2,
            payload={"good": "iron"},
        )


def test_proposals_about_other_goods_or_to_other_agents_stay_open(bus: MessageBus) -> None:
    _propose(bus, MessageType.OFFER, good="iron")
    _propose(bus, MessageType.REQUEST, good="wood")
    _propose(bus, MessageType.OFFER, good="iron", recipient="woodworking_1")
    # The other direction is its own proposal too.
    _propose(bus, MessageType.REQUEST, good="iron", sender="smithing_1", recipient="warrior_1")

    assert bus.open_proposal("smithing_1", "warrior_1", "iron") is MessageType.OFFER
    assert bus.open_proposal("smithing_1", "warrior_1", "wood") is MessageType.REQUEST
    assert bus.open_proposal("woodworking_1", "warrior_1", "iron") is MessageType.OFFER
    assert bus.open_proposal("warrior_1", "smithing_1", "iron") is MessageType.REQUEST


def test_a_replacing_proposal_expires_on_its_own_clock() -> None:
    bus, _ = expiring_bus(ticks=6)
    offer_iron(bus, tick=0)
    bus.send(
        sender_id="warrior_1",
        recipient_id="smithing_1",
        message_type=MessageType.REQUEST,
        tick=4,
        payload={"good": "iron", "quantity": 1, "price": "10"},
    )
    assert bus.open_proposal("smithing_1", "warrior_1", "iron", tick=9) is MessageType.REQUEST
    assert bus.open_proposal("smithing_1", "warrior_1", "iron", tick=10) is None


def test_an_ack_after_an_offer_then_a_request_answers_the_request(tmp_path) -> None:
    """Accepting the latest proposal, a REQUEST, places a sell, not a buy."""
    from rpg_economy.agents.action import AgentAction
    from rpg_economy.agents.executor import execute
    from rpg_economy.domain import ActionType, Condition, OrderSide
    from rpg_economy.engine import Engine, build_world
    from rpg_economy.logging_ import EventLogWriter
    from rpg_economy.orchestration import DecentralizedOrchestrator, build_directory
    from tests.conftest import make_config, make_metadata

    config = make_config(run={"condition": "B"})
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        messages = MessageBus(build_directory(engine))
        DecentralizedOrchestrator(config, engine, lambda e, a: None, messages=messages)
        engine.state.account("warrior_1").receive(parse_good("iron"), 2)
        engine.state.account("smithing_1").receive(parse_good("iron"), 2)

        def propose(kind: MessageType) -> None:
            action = AgentAction(
                action=ActionType.SEND_MESSAGE,
                message_type=kind,
                target="iron",
                quantity=1,
                price="10",
                recipient="smithing_1",
            )
            assert execute(
                engine, "warrior_1", action, messages=messages, condition=Condition.B
            ).accepted

        propose(MessageType.OFFER)
        propose(MessageType.REQUEST)

        ack = AgentAction(
            action=ActionType.SEND_MESSAGE,
            message_type=MessageType.ACK,
            target="iron",
            quantity=1,
            price="10",
            recipient="warrior_1",
        )
        assert execute(engine, "smithing_1", ack, messages=messages, condition=Condition.B).accepted
        (order,) = engine.market.open_orders_of("smithing_1")
        assert order.side is OrderSide.SELL
        assert messages.open_proposal("smithing_1", "warrior_1", "iron") is None


def test_send_returns_the_delivered_message_with_its_id(bus: MessageBus) -> None:
    """The executor logs message_id from what send returns."""
    kwargs = {"sender_id": "woodworking_1", "recipient_id": "warrior_1", "tick": 1}
    first = bus.send(message_type=MessageType.OFFER, payload={"good": "iron"}, **kwargs)
    second = bus.send(message_type=MessageType.INFORM, payload={"good": "wood"}, **kwargs)
    assert second.message_id == first.message_id + 1
    assert bus.inbox("warrior_1") == [first, second]
    assert first.to_payload()["message_id"] == first.message_id


# ---------------------------------------------------------------------------
# Listing the proposals still open to an agent
# ---------------------------------------------------------------------------
def test_open_proposals_list_what_can_still_be_answered_with_ticks_left() -> None:
    bus, clock = expiring_bus(ticks=6)
    offer_iron(bus, tick=0)
    bus.send(
        sender_id="warrior_1",
        recipient_id="smithing_1",
        message_type=MessageType.REQUEST,
        tick=2,
        payload={"good": "wood", "quantity": 2, "price": "4"},
    )
    bus.send(
        sender_id="warrior_1",
        recipient_id="smithing_1",
        message_type=MessageType.INFORM,
        tick=2,
        payload={"good": "herbs"},
    )
    bus.drain("smithing_1")  # the inbox empties; the proposals stay open
    clock[0] = 3

    listed = [(m.type, m.payload["good"], left) for m, left in bus.open_proposals("smithing_1")]
    assert listed == [(MessageType.OFFER, "iron", 3), (MessageType.REQUEST, "wood", 5)]
    assert bus.open_proposals("warrior_1") == []

    clock[0] = 6  # the offer has expired, the request has not
    assert [m.payload["good"] for m, _ in bus.open_proposals("smithing_1")] == ["wood"]
    assert bus.open_proposal("smithing_1", "warrior_1", "iron") is None


def test_an_answered_or_replaced_proposal_is_not_listed() -> None:
    bus, clock = expiring_bus(ticks=6)
    offer_iron(bus, tick=0)
    offer_iron(bus, tick=1)  # replaces the first
    assert [m.message_id for m, _ in bus.open_proposals("smithing_1")] == [2]
    bus.send(
        sender_id="smithing_1",
        recipient_id="warrior_1",
        message_type=MessageType.REJECT,
        tick=1,
        payload={"good": "iron"},
    )
    assert bus.open_proposals("smithing_1") == []


def test_without_an_expiry_open_proposals_have_no_ticks_left(bus: MessageBus) -> None:
    bus.send(
        sender_id="warrior_1",
        recipient_id="smithing_1",
        message_type=MessageType.OFFER,
        tick=0,
        payload={"good": "iron"},
    )
    ((message, left),) = bus.open_proposals("smithing_1")
    assert message.type is MessageType.OFFER
    assert left is None
