"""Condition B end to end, fully offline (phase 3 gate).

Includes the T1 integration requirement: an LLM agent emits a peer-to-peer
message, and the recipient's next activation is served on a logged T1 firing.
Rule-based agents never message anyone, so this is T1's only end-to-end
coverage.
"""

from __future__ import annotations

import collections
from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.agents.offline import agent_id_from_prompt, json_action
from rpg_economy.agents.prompts import PROMPT_VERSION
from rpg_economy.domain import ActionType, Condition, MessageType, Product, TokenBucket, Trigger
from rpg_economy.llm import MockLLMClient, always
from rpg_economy.logging_ import EventType, log_digest, read_events
from rpg_economy.runner import run_decentralized
from tests.conftest import gold_flow, make_config

CYCLES = 60


def counts(path: Path) -> collections.Counter:
    return collections.Counter(e.type.value for e in read_events(path))


# ---------------------------------------------------------------------------
# The T1 integration gate
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_p2p_message_triggers_the_recipients_next_activation(tmp_path: Path) -> None:
    """Message sent by an LLM agent -> logged T1 firing -> recipient served.

    Run with K=1 so exactly one agent acts per macro-cycle. At the default K
    most agents are served every cycle, and a recipient scheduled *after* the
    sender in the same cycle simply reads the mail in that activation -- correct
    behaviour, but it leaves T1 with nothing to do and the causal chain
    unobservable. One activation per cycle forces the message to wait for a
    later cycle, which is exactly the case T1 exists to cover.
    """
    config = make_config(run={"condition": "B"}, time={"k_activations": 1})
    sent = {"done": False}

    def respond(system: str, user: str) -> str:
        if agent_id_from_prompt(user) == "woodworking_1" and not sent["done"]:
            sent["done"] = True
            return json_action(
                action="send_message",
                message_type=MessageType.REQUEST.value,
                target="iron",
                quantity=2,
                price="5",
                recipient="warrior_1",
                short_reason="ask a peer for iron",
            )
        return json_action(action="wait", short_reason="idle")

    result = run_decentralized(config, tmp_path / "run.jsonl", responder=respond, cycles=40)
    events = list(read_events(result.log_path))

    sends = [e for e in events if e.type is EventType.MESSAGE_SENT]
    assert sends, "no message was ever sent"
    send = sends[0]
    assert send.agent_id == "woodworking_1"
    assert send.payload["recipient_id"] == "warrior_1"

    t1 = [
        e
        for e in events
        if e.type is EventType.TRIGGER_FIRED
        and e.agent_id == "warrior_1"
        and e.payload["trigger"] == Trigger.T1_MESSAGE_RECEIVED.value
        and e.seq > send.seq
    ]
    assert t1, "the recipient's T1 never fired after the message"

    served = [
        e
        for e in events
        if e.type is EventType.ACTIVATION_SERVED
        and e.agent_id == "warrior_1"
        and e.seq > t1[0].seq
        and Trigger.T1_MESSAGE_RECEIVED.value in e.payload["triggers"]
    ]
    assert served, "the recipient was never activated on its T1"


@pytest.mark.invariant
def test_the_recipient_actually_reads_the_message(tmp_path: Path) -> None:
    """The message must reach the prompt, not merely trigger an activation."""
    config = make_config(run={"condition": "B"}, time={"k_activations": 1})
    prompts: list[str] = []
    sent = {"done": False}

    def respond(system: str, user: str) -> str:
        prompts.append(user)
        if agent_id_from_prompt(user) == "woodworking_1" and not sent["done"]:
            sent["done"] = True
            return json_action(
                action="send_message",
                message_type=MessageType.INFORM.value,
                target="iron",
                recipient="warrior_1",
                short_reason="tell a peer",
            )
        return json_action(action="wait")

    run_decentralized(config, tmp_path / "run.jsonl", responder=respond, cycles=40)

    assert any("Messages addressed to you" in p for p in prompts), "no prompt ever carried an inbox"


@pytest.mark.invariant
def test_p2p_content_is_billed_when_a_message_is_read(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    result = run_decentralized(config, tmp_path / "run.jsonl", cycles=CYCLES)

    events = list(read_events(result.log_path))
    assert any(e.type is EventType.MESSAGE_SENT for e in events)
    assert result.ledger.input_tokens(TokenBucket.P2P_CONTENT) > 0


# ---------------------------------------------------------------------------
# A complete offline run
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_condition_b_runs_fully_offline(tmp_path: Path) -> None:
    """No network, no credentials, complete pipeline."""
    config = make_config(run={"condition": "B"})
    result = run_decentralized(config, tmp_path / "run.jsonl", cycles=CYCLES)

    assert isinstance(result.client, MockLLMClient)
    assert result.client.calls > 0
    assert result.ledger.total() > 0


@pytest.mark.invariant
def test_a_full_offline_run_is_deterministic(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    first = run_decentralized(config, tmp_path / "a.jsonl", cycles=CYCLES)
    second = run_decentralized(config, tmp_path / "b.jsonl", cycles=CYCLES)

    assert log_digest(first.log_path) == log_digest(second.log_path)
    assert first.ledger.snapshot() == second.ledger.snapshot()


def test_the_offline_run_exercises_the_whole_pipeline(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    result = run_decentralized(config, tmp_path / "run.jsonl", cycles=CYCLES)
    seen = counts(result.log_path)

    for required in (
        EventType.LLM_CALL,
        EventType.AGENT_OUTPUT,
        EventType.ACTION_EXECUTED,
        EventType.TRIGGER_FIRED,
        EventType.ACTIVATION_SERVED,
        EventType.MESSAGE_SENT,
        EventType.TRADE_EXECUTED,
        EventType.TOKENS_CONSUMED,
    ):
        assert seen.get(required.value, 0) > 0, f"the run never produced {required}"


def test_run_metadata_records_the_prompt_hashes(tmp_path: Path) -> None:
    """Design v1.0 s15: a run is tied to the wording that produced it."""
    from rpg_economy.logging_ import read_metadata

    config = make_config(run={"condition": "B"})
    result = run_decentralized(config, tmp_path / "run.jsonl", cycles=5)
    metadata = read_metadata(result.log_path)

    assert metadata.prompt_hashes
    # Tied to PROMPT_VERSION rather than a literal: the point is that metadata
    # records the wording that ran, not that the version never moves.
    assert metadata.prompt_hashes["version"] == PROMPT_VERSION
    assert "global_view_format" in metadata.prompt_hashes
    # The format version of the view the Director and the bulletin share.
    from rpg_economy.orchestration.global_view import GLOBAL_VIEW_VERSION

    assert metadata.global_view_version == GLOBAL_VIEW_VERSION
    assert metadata.llm_mock is True


@pytest.mark.invariant
def test_a_full_offline_run_conserves_gold(tmp_path: Path) -> None:
    from rpg_economy.engine import build_world

    config = make_config(run={"condition": "B"})
    expected = build_world(config).total_gold()
    result = run_decentralized(config, tmp_path / "run.jsonl", cycles=CYCLES)
    assert result.state.total_gold() == expected + gold_flow(result.log_path)


@pytest.mark.invariant
def test_k_is_respected_across_a_full_offline_run(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    result = run_decentralized(config, tmp_path / "run.jsonl", cycles=CYCLES)

    for event in read_events(result.log_path):
        if event.type is EventType.CYCLE_END:
            assert event.payload["effective_activations"] <= config.time.k_activations


def test_token_totals_are_logged_at_the_end_of_a_run(tmp_path: Path) -> None:
    config = make_config(run={"condition": "B"})
    result = run_decentralized(config, tmp_path / "run.jsonl", cycles=20)

    totals = [e for e in read_events(result.log_path) if e.type is EventType.TOKENS_CONSUMED]
    assert len(totals) == 1
    payload = totals[0].payload
    assert payload["grand_total"] == result.ledger.total()
    assert sum(payload["total"].values()) == payload["grand_total"]


def test_a_run_that_always_fails_to_parse_still_completes(tmp_path: Path) -> None:
    """Retry-then-WAIT must degrade gracefully, not abort the run."""
    config = make_config(run={"condition": "B"})
    result = run_decentralized(
        config, tmp_path / "run.jsonl", responder=always("not json"), cycles=10
    )

    seen = counts(result.log_path)
    assert seen[EventType.ACTIVATION_WASTED.value] > 0
    assert seen[EventType.AGENT_OUTPUT_INVALID.value] > 0
    assert result.state.tick > 0


def test_a_trio_seed_shares_world_randomness(tmp_path: Path) -> None:
    """A and B on one seed must see the same shock at the same tick."""
    shocks = {}
    for condition in (Condition.B, Condition.C):
        config = make_config(run={"condition": condition.value})
        result = run_decentralized(config, tmp_path / f"{condition.value}.jsonl", cycles=CYCLES)
        shocks[condition] = [
            (e.type, e.tick)
            for e in read_events(result.log_path)
            if e.type in {EventType.SHOCK_APPLIED, EventType.SHOCK_REVERTED}
        ]
    assert shocks[Condition.B] == shocks[Condition.C]


# ---------------------------------------------------------------------------
# The offline mock's peer replies: no acknowledgment ping-pong
# ---------------------------------------------------------------------------
def _mock_rig(tmp_path: Path, messages_in: list[tuple[str, MessageType, dict]]):
    """woodworking_1's prompt with ``messages_in`` in its inbox, plus engine and bus.

    woodworking_1 is given two health potions to trade. An answer (ACK or REJECT)
    needs a proposal to answer, so woodworking_1 first sends the peer an OFFER of
    those two potions at 18.
    """
    from rpg_economy.agents.observation import build_observation
    from rpg_economy.agents.prompts import build_agent_prompt
    from rpg_economy.engine import Engine, build_world
    from rpg_economy.logging_ import EventLogWriter
    from rpg_economy.messaging import MessageBus
    from rpg_economy.orchestration import build_directory
    from tests.conftest import make_metadata

    config = make_config(run={"condition": "B"})
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        engine.state.account("woodworking_1").receive(Product.HEALTH_POTION, 2)
        bus = MessageBus(build_directory(engine))
        for sender, kind, payload in messages_in:
            if kind in (MessageType.ACK, MessageType.REJECT):
                bus.send(
                    sender_id="woodworking_1",
                    recipient_id=sender,
                    message_type=MessageType.OFFER,
                    tick=0,
                    payload={"good": payload["good"], "quantity": 2, "price": "18"},
                )
            bus.send(
                sender_id=sender,
                recipient_id="woodworking_1",
                message_type=kind,
                tick=0,
                payload=payload,
            )
        observation = build_observation(config, engine, "woodworking_1", messages=bus)
        prompt = build_agent_prompt(observation, config).user
    return prompt, engine, bus


@pytest.mark.invariant
@pytest.mark.parametrize("closing", [MessageType.ACK, MessageType.REJECT])
def test_the_mock_never_answers_an_ack_or_reject(tmp_path: Path, closing: MessageType) -> None:
    from rpg_economy.agents.offline import _inbox, _reply_to_sender

    prompt, engine, _ = _mock_rig(tmp_path, [("warrior_1", closing, {"good": "health_potion"})])
    assert "Messages addressed to you" in prompt
    assert _reply_to_sender(engine, "woodworking_1", _inbox(prompt)) is None


def test_the_mock_accepts_a_fair_offer_and_declines_a_dear_one(tmp_path: Path) -> None:
    from rpg_economy.agents.offline import _inbox, _reply_to_sender

    for price, expected in (("9", MessageType.ACK), ("90", MessageType.REJECT)):
        prompt, engine, _ = _mock_rig(
            tmp_path / price,
            [("warrior_1", MessageType.OFFER, {"good": "iron", "quantity": 2, "price": price})],
        )
        reply = _reply_to_sender(engine, "woodworking_1", _inbox(prompt))
        assert reply is not None
        assert reply.action is ActionType.SEND_MESSAGE
        assert reply.message_type is expected
        assert reply.recipient == "warrior_1"
        assert reply.target == "iron"


def test_the_mock_accepts_a_request_only_for_stock_it_holds(tmp_path: Path) -> None:
    from rpg_economy.agents.offline import _inbox, _reply_to_sender

    # The rig gives woodworking_1 health potions; it holds no iron.
    for good, price, expected in (
        ("health_potion", "18", MessageType.ACK),
        ("iron", "10", MessageType.REJECT),
    ):
        prompt, engine, _ = _mock_rig(
            tmp_path / good,
            [("warrior_1", MessageType.REQUEST, {"good": good, "quantity": 1, "price": price})],
        )
        reply = _reply_to_sender(engine, "woodworking_1", _inbox(prompt))
        assert reply is not None and reply.message_type is expected


def test_an_accepted_offer_is_placed_on_the_book(tmp_path: Path) -> None:
    from rpg_economy.agents.offline import _follow_up, _inbox

    prompt, engine, bus = _mock_rig(
        tmp_path, [("warrior_1", MessageType.ACK, {"good": "health_potion"})]
    )
    action = _follow_up(engine, bus, "woodworking_1", _inbox(prompt))
    assert action is not None
    assert action.action is ActionType.SELL
    assert action.target == "health_potion"
    assert action.quantity == 2
    assert action.price == Decimal("18.00")


def test_a_rejected_offer_is_placed_cheaper(tmp_path: Path) -> None:
    from rpg_economy.agents.offline import _follow_up, _inbox

    prompt, engine, bus = _mock_rig(
        tmp_path, [("warrior_1", MessageType.REJECT, {"good": "health_potion"})]
    )
    action = _follow_up(engine, bus, "woodworking_1", _inbox(prompt))
    assert action is not None
    assert action.action is ActionType.SELL
    assert action.price == Decimal("17.10")


def test_an_empty_inbox_gets_no_reply() -> None:
    from rpg_economy.agents.offline import _inbox

    assert _inbox("no inbox in this prompt") == []


def test_the_mock_offers_only_to_agents_that_use_the_good(tmp_path: Path) -> None:
    """Herbs go to alchemists, a weapon to warriors: never to an agent with no use for it."""
    from rpg_economy.agents.offline import _pick_peer
    from rpg_economy.engine import Engine, build_world
    from rpg_economy.logging_ import EventLogWriter
    from tests.conftest import make_metadata

    config = make_config(run={"condition": "B"})
    expected = {
        "herbs": {"alchemy"},
        "iron": {"smithing"},
        "wood": {"woodworking"},
        "melee_weapon": {"warrior"},
        "armor": {"warrior", "ranger", "mage"},
        "mana_potion": {"mage"},
    }
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        for good, users in expected.items():
            for _ in range(20):
                peer = _pick_peer(config, engine, "woodworking_1", good, selling=True)
                assert peer is not None
                assert engine.state.account(peer).specialization in users, (good, peer)


def test_the_mock_asks_only_agents_that_supply_the_good(tmp_path: Path) -> None:
    from rpg_economy.agents.offline import _pick_peer
    from rpg_economy.engine import Engine, build_world
    from rpg_economy.logging_ import EventLogWriter
    from tests.conftest import make_metadata

    config = make_config(run={"condition": "B"})
    expected = {
        "health_potion": {"alchemy"},
        "bow_gear": {"woodworking"},
        "iron": {"warrior"},
        "arcane_crystal": {"mage"},
    }
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        for good, suppliers in expected.items():
            for _ in range(20):
                peer = _pick_peer(config, engine, "woodworking_1", good, selling=False)
                assert peer is not None
                assert engine.state.account(peer).specialization in suppliers, (good, peer)


def test_accepting_an_offer_places_the_buy_in_the_same_action(tmp_path: Path) -> None:
    from rpg_economy.agents.offline import _inbox, _reply_to_sender

    prompt, engine, _ = _mock_rig(
        tmp_path,
        [("warrior_1", MessageType.OFFER, {"good": "iron", "quantity": 2, "price": "9"})],
    )
    reply = _reply_to_sender(engine, "woodworking_1", _inbox(prompt))
    assert reply is not None and reply.message_type is MessageType.ACK
    assert reply.quantity == 2
    assert reply.price == Decimal("9.00")


@pytest.mark.invariant
def test_no_agent_acknowledges_more_than_it_was_offered_or_asked(tmp_path: Path) -> None:
    """The ACK ping-pong: each acknowledgment fired the other side's T1."""
    config = make_config(run={"condition": "B"})
    result = run_decentralized(config, tmp_path / "run.jsonl", cycles=CYCLES)

    received: collections.Counter = collections.Counter()
    acks: collections.Counter = collections.Counter()
    for event in read_events(result.log_path):
        if event.type is not EventType.MESSAGE_SENT:
            continue
        kind = event.payload["type"]
        if kind in (MessageType.REQUEST.value, MessageType.OFFER.value):
            received[event.payload["recipient_id"]] += 1
        elif kind == MessageType.ACK.value:
            acks[event.agent_id] += 1

    assert sum(received.values()) > 0, "the run sent no offers, so this proves nothing"
    for agent_id, sent in acks.items():
        assert sent <= received[agent_id], f"{agent_id} sent {sent} ACKs for {received[agent_id]}"
