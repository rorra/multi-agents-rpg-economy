"""Mechanical triggers T1-T5 (design v1.0 s8.2)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.domain import MessageType, OrderSide, Product, Resource, Trigger
from rpg_economy.engine import Engine, build_world
from rpg_economy.logging_ import EventLogWriter, EventType, read_events
from rpg_economy.messaging import MessageBus
from rpg_economy.orchestration import TriggerEvaluator, build_directory
from tests.conftest import make_config, make_metadata

IRON = Resource.IRON


@pytest.fixture
def rig(tmp_path: Path):
    """An engine in Condition B with a trigger evaluator attached."""
    config = make_config(run={"condition": "B"})
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        messages = MessageBus(build_directory(engine))
        evaluator = TriggerEvaluator(config, engine, messages)
        # Start from a clean slate: everyone just acted.
        for agent_id in state.agent_ids:
            evaluator.on_activated(agent_id, tick=0)
        yield config, engine, messages, evaluator


def triggers_for(evaluator: TriggerEvaluator, agent_id: str) -> set[Trigger]:
    request = evaluator.pending.get(agent_id)
    return set(request.triggers) if request else set()


# ---------------------------------------------------------------------------
# T1 -- peer message received
# ---------------------------------------------------------------------------
def test_t1_fires_on_a_received_message(rig) -> None:
    _, engine, messages, evaluator = rig
    messages.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.REQUEST,
        tick=1,
    )
    evaluator.evaluate(1)
    assert Trigger.T1_MESSAGE_RECEIVED in triggers_for(evaluator, "warrior_1")


def test_t1_does_not_fire_for_the_sender(rig) -> None:
    _, engine, messages, evaluator = rig
    messages.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.OFFER,
        tick=1,
    )
    evaluator.evaluate(1)
    assert Trigger.T1_MESSAGE_RECEIVED not in triggers_for(evaluator, "woodworking_1")


def test_t1_clears_once_the_agent_is_activated(rig) -> None:
    _, engine, messages, evaluator = rig
    messages.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.INFORM,
        tick=1,
    )
    evaluator.evaluate(1)
    evaluator.on_activated("warrior_1", tick=1)
    evaluator.evaluate(2)
    assert Trigger.T1_MESSAGE_RECEIVED not in triggers_for(evaluator, "warrior_1")


# ---------------------------------------------------------------------------
# The bulletin fires nothing (design v1.0 s3.3)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_the_bulletin_never_fires_t1(rig) -> None:
    """Condition C's bulletin is engine information, not a message."""
    _, engine, messages, evaluator = rig
    engine.publish_bulletin({"prices": {"iron": "10.00"}, "volume": {"iron": 5}})
    evaluator.evaluate(1)

    for agent_id in engine.state.agent_ids:
        assert Trigger.T1_MESSAGE_RECEIVED not in triggers_for(evaluator, agent_id)


@pytest.mark.invariant
def test_the_bulletin_creates_no_message(rig) -> None:
    """Structural: it never enters the bus, so it cannot count as P2P traffic."""
    _, engine, messages, evaluator = rig
    engine.publish_bulletin({"prices": {}})

    assert messages.sent == []
    for agent_id in engine.state.agent_ids:
        assert messages.unread_count(agent_id) == 0


@pytest.mark.invariant
def test_the_bulletin_fires_no_trigger_at_all(rig) -> None:
    """s3.3: it fires neither T1 nor any other trigger."""
    _, engine, messages, evaluator = rig
    evaluator.evaluate(0)  # establish whatever fires on its own
    before = {a: triggers_for(evaluator, a) for a in engine.state.agent_ids}

    engine.publish_bulletin({"prices": {"iron": "10.00"}})
    evaluator.evaluate(0)

    for agent_id in engine.state.agent_ids:
        assert triggers_for(evaluator, agent_id) == before[agent_id]


def test_the_bulletin_is_logged_as_engine_output(rig) -> None:
    _, engine, _, _ = rig
    engine.publish_bulletin({"prices": {"iron": "10.00"}})
    engine.log.close()

    types = [e.type for e in read_events(engine.log.path)]
    assert EventType.BULLETIN_PUBLISHED in types


# ---------------------------------------------------------------------------
# T2 -- own order event
# ---------------------------------------------------------------------------
def test_t2_fires_when_an_order_is_filled(rig) -> None:
    _, engine, _, evaluator = rig
    engine.state.account("warrior_1").receive(IRON, 5)
    engine.submit_order("warrior_1", OrderSide.SELL, IRON, 2, "10")
    engine.state.account("woodworking_1").receive(IRON, 0)
    engine.submit_order("woodworking_1", OrderSide.BUY, IRON, 2, "10")

    evaluator.evaluate(1)
    assert Trigger.T2_ORDER_EVENT in triggers_for(evaluator, "warrior_1")
    assert Trigger.T2_ORDER_EVENT in triggers_for(evaluator, "woodworking_1")


def test_t2_fires_on_expiry(rig) -> None:
    config, engine, _, evaluator = rig
    engine.state.account("warrior_1").receive(IRON, 5)
    engine.submit_order("warrior_1", OrderSide.SELL, IRON, 2, "10")

    engine.state.tick = config.market.order_expiry_ticks
    engine.market.expire_due(engine.tick)
    evaluator.evaluate(engine.tick)

    assert Trigger.T2_ORDER_EVENT in triggers_for(evaluator, "warrior_1")


def test_t2_fires_on_cancellation(rig) -> None:
    _, engine, _, evaluator = rig
    engine.state.account("warrior_1").receive(IRON, 5)
    order = engine.submit_order("warrior_1", OrderSide.SELL, IRON, 2, "10")
    engine.market.cancel(order.order_id)

    evaluator.evaluate(1)
    assert Trigger.T2_ORDER_EVENT in triggers_for(evaluator, "warrior_1")


def test_placing_an_order_alone_does_not_fire_t2(rig) -> None:
    """T2 is about an order's *outcome*, not about having placed one."""
    _, engine, _, evaluator = rig
    engine.state.account("warrior_1").receive(IRON, 5)
    engine.submit_order("warrior_1", OrderSide.SELL, IRON, 2, "99")

    evaluator.evaluate(1)
    assert Trigger.T2_ORDER_EVENT not in triggers_for(evaluator, "warrior_1")


# ---------------------------------------------------------------------------
# T3 -- price move in the agent's fixed interest list
# ---------------------------------------------------------------------------
def trade_iron(engine: Engine, price: str, quantity: int = 1) -> None:
    engine.state.account("ranger_2").receive(IRON, quantity)
    engine.submit_order("ranger_2", OrderSide.SELL, IRON, quantity, price)
    engine.submit_order("woodworking_2", OrderSide.BUY, IRON, quantity, price)


def test_t3_fires_when_a_watched_price_moves(rig) -> None:
    config, engine, _, evaluator = rig
    trade_iron(engine, "10")
    for agent_id in engine.state.agent_ids:
        evaluator.on_activated(agent_id, tick=1)

    trade_iron(engine, "20")  # well beyond the threshold
    evaluator.evaluate(2)

    assert Trigger.T3_PRICE_MOVE in triggers_for(evaluator, "warrior_1")


def test_t3_ignores_a_move_below_the_threshold(rig) -> None:
    config, engine, _, evaluator = rig
    trade_iron(engine, "10")
    for agent_id in engine.state.agent_ids:
        evaluator.on_activated(agent_id, tick=1)

    small = (Decimal("10") * (Decimal(1) + config.triggers.t3_price_move_pct / 2)).quantize(
        Decimal("0.01")
    )
    trade_iron(engine, str(small))
    evaluator.evaluate(2)

    assert Trigger.T3_PRICE_MOVE not in triggers_for(evaluator, "warrior_1")


def test_t3_only_watches_goods_in_the_interest_list(rig) -> None:
    """A woodworker does not follow iron (design v1.0 s5.2)."""
    config, engine, _, evaluator = rig
    assert "iron" not in config.observability.interest_of("artisan", "woodworking")

    trade_iron(engine, "10")
    for agent_id in engine.state.agent_ids:
        evaluator.on_activated(agent_id, tick=1)
    trade_iron(engine, "40")
    evaluator.evaluate(2)

    assert Trigger.T3_PRICE_MOVE not in triggers_for(evaluator, "woodworking_1")
    assert Trigger.T3_PRICE_MOVE in triggers_for(evaluator, "smithing_1")


# ---------------------------------------------------------------------------
# T4 -- own-state threshold
# ---------------------------------------------------------------------------
def test_t4_fires_on_low_gold(rig) -> None:
    config, engine, _, evaluator = rig
    engine.state.account("warrior_1").gold = config.triggers.t4_gold_low - 1
    evaluator.evaluate(1)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")


def test_t4_fires_on_high_gold(rig) -> None:
    config, engine, _, evaluator = rig
    engine.state.account("warrior_1").gold = config.triggers.t4_gold_high + 1
    evaluator.evaluate(1)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")


def _settle_gold(config, engine: Engine, evaluator: TriggerEvaluator, tick: int = 1):
    """Put warrior_1 between the gold bounds and clear its trigger state."""
    account = engine.state.account("warrior_1")
    account.gold = (config.triggers.t4_gold_low + config.triggers.t4_gold_high) / 2
    evaluator.evaluate(tick)
    evaluator.on_activated("warrior_1", tick)
    return account


def test_t4_gold_low_reads_available_gold_so_a_freeze_is_visible(rig) -> None:
    """Frozen gold cannot be spent: a liquidity freeze is a low-gold crossing."""
    config, engine, _, evaluator = rig
    account = _settle_gold(config, engine, evaluator)
    account.frozen_gold = account.gold - config.triggers.t4_gold_low + 1
    evaluator.evaluate(2)
    assert account.gold > config.triggers.t4_gold_low
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")


def test_t4_gold_low_counts_gold_reserved_by_open_bids_as_unavailable(rig) -> None:
    config, engine, _, evaluator = rig
    account = _settle_gold(config, engine, evaluator)
    account.reserved_gold = account.gold - config.triggers.t4_gold_low
    evaluator.evaluate(2)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")


def test_t4_gold_high_reads_available_gold(rig) -> None:
    config, engine, _, evaluator = rig
    account = _settle_gold(config, engine, evaluator)
    account.gold = config.triggers.t4_gold_high + 10
    account.frozen_gold = Decimal(20)
    evaluator.evaluate(2)
    assert "gold_high" not in evaluator._t4_conditions("warrior_1", 2)
    assert Trigger.T4_OWN_STATE_THRESHOLD not in triggers_for(evaluator, "warrior_1")
    account.frozen_gold = Decimal(0)
    evaluator.evaluate(3)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")


@pytest.mark.invariant
def test_the_t4_gold_rule_is_identical_in_b_and_c(tmp_path: Path) -> None:
    """Same state, same thresholds, same conditions: the bulletin changes nothing."""
    held = {}
    for condition in ("B", "C"):
        config = make_config(run={"condition": condition})
        with EventLogWriter(tmp_path / f"{condition}.jsonl", make_metadata(config)) as log:
            engine = Engine(config, build_world(config), log)
            evaluator = TriggerEvaluator(config, engine, MessageBus(build_directory(engine)))
            account = engine.state.account("warrior_1")
            account.gold = config.triggers.t4_gold_high + 5
            account.frozen_gold = account.gold - config.triggers.t4_gold_low
            held[condition] = evaluator._t4_conditions("warrior_1", 1)
    assert "gold_low" in held["B"] and "gold_high" not in held["B"]
    assert held["B"] == held["C"]


def test_t4_fires_on_critical_durability(rig) -> None:
    config, engine, _, evaluator = rig
    account = engine.state.account("warrior_1")
    account.gold = Decimal("100")
    full = config.economy.durability[Product.MELEE_WEAPON]
    account.equipment[Product.MELEE_WEAPON] = 1  # the last expedition left
    assert Decimal(1) / full <= config.triggers.t4_durability_critical_pct

    evaluator.evaluate(1)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")


def test_t4_fires_when_recovery_finishes(rig) -> None:
    config, engine, _, evaluator = rig
    account = engine.state.account("warrior_1")
    account.gold = Decimal("100")
    account.recovering_until_tick = 5
    evaluator.on_activated("warrior_1", tick=3)  # snapshot: still recovering
    assert Trigger.T4_OWN_STATE_THRESHOLD not in triggers_for(evaluator, "warrior_1")

    engine.state.tick = 6
    evaluator.evaluate(6)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")


def test_t4_fires_when_an_adventurer_is_back_in_town(rig) -> None:
    """Back from a trip it has a haul to sell and gear to replace, and only its rest."""
    config, engine, _, evaluator = rig
    account = engine.state.account("warrior_1")
    account.gold = Decimal("100")
    engine.start_expedition("warrior_1", "green_hollow")
    evaluator.on_activated("warrior_1", tick=0)  # snapshot: away
    assert Trigger.T4_OWN_STATE_THRESHOLD not in triggers_for(evaluator, "warrior_1")

    account.expedition = None
    account.recovering_until_tick = 10  # resting: not yet available
    engine.state.tick = 4
    evaluator.evaluate(4)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")


def test_t4_fires_for_a_blocked_artisan(rig) -> None:
    _, engine, _, evaluator = rig
    account = engine.state.account("smithing_1")
    account.gold = Decimal("100")
    account.inventory.clear()
    evaluator.evaluate(1)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "smithing_1")


@pytest.mark.invariant
def test_t4_fires_on_the_crossing_not_while_the_condition_holds(rig) -> None:
    """Design s8.2: a threshold *crossing*. A held condition is not a new one."""
    config, engine, _, evaluator = rig
    account = engine.state.account("warrior_1")
    account.gold = config.triggers.t4_gold_low - 1
    evaluator.evaluate(1)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")

    evaluator.on_activated("warrior_1", tick=1)
    for tick in range(2, 4):
        evaluator.evaluate(tick)
    assert Trigger.T4_OWN_STATE_THRESHOLD not in triggers_for(evaluator, "warrior_1")

    # Back inside the bounds, then below again: a second crossing.
    account.gold = Decimal("100")
    evaluator.evaluate(4)
    account.gold = config.triggers.t4_gold_low - 1
    evaluator.evaluate(5)
    assert Trigger.T4_OWN_STATE_THRESHOLD in triggers_for(evaluator, "warrior_1")


def test_an_artisan_does_not_ask_to_act_every_tick(rig) -> None:
    """Idle or blocked is the artisan's normal state; only a change fires T4."""
    _, engine, _, evaluator = rig
    engine.state.account("smithing_1").gold = Decimal("100")
    for tick in range(1, 4):
        evaluator.evaluate(tick)
    assert Trigger.T4_OWN_STATE_THRESHOLD not in triggers_for(evaluator, "smithing_1")


def test_a_condition_already_true_at_the_start_is_not_a_crossing(rig) -> None:
    config, engine, messages, _ = rig
    engine.state.account("warrior_1").gold = config.triggers.t4_gold_low - 1
    fresh = TriggerEvaluator(config, engine, messages)
    fresh.evaluate(1)
    assert Trigger.T4_OWN_STATE_THRESHOLD not in triggers_for(fresh, "warrior_1")


# ---------------------------------------------------------------------------
# T5 -- inactivity
# ---------------------------------------------------------------------------
def test_t5_fires_after_n_ticks_of_inactivity(rig) -> None:
    config, engine, _, evaluator = rig
    n = config.triggers.t5_inactivity_ticks

    evaluator.evaluate(n - 1)
    assert Trigger.T5_INACTIVITY not in triggers_for(evaluator, "warrior_1")

    evaluator.evaluate(n)
    assert Trigger.T5_INACTIVITY in triggers_for(evaluator, "warrior_1")


def test_t5_resets_on_activation(rig) -> None:
    config, engine, _, evaluator = rig
    n = config.triggers.t5_inactivity_ticks
    evaluator.evaluate(n)
    evaluator.on_activated("warrior_1", tick=n)

    evaluator.evaluate(n + 1)
    assert Trigger.T5_INACTIVITY not in triggers_for(evaluator, "warrior_1")


# ---------------------------------------------------------------------------
# Persistence and logging
# ---------------------------------------------------------------------------
def test_an_unserved_request_persists_while_its_trigger_holds(rig) -> None:
    """Design v1.0 s8.3: unserved requests stand while the condition holds."""
    _, engine, messages, evaluator = rig
    messages.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.REQUEST,
        tick=1,
    )
    evaluator.evaluate(1)
    first = evaluator.pending["warrior_1"].first_requested_tick

    for tick in range(2, 6):
        evaluator.evaluate(tick)
        evaluator.drop_resolved(tick)

    assert "warrior_1" in evaluator.pending
    assert evaluator.pending["warrior_1"].first_requested_tick == first


def test_a_resolved_state_trigger_is_dropped(rig) -> None:
    """Only T4 can resolve without the agent acting."""
    config, engine, _, evaluator = rig
    account = engine.state.account("warrior_1")
    account.gold = config.triggers.t4_gold_low - 1
    evaluator.evaluate(1)
    assert "warrior_1" in evaluator.pending

    account.gold = (config.triggers.t4_gold_low + config.triggers.t4_gold_high) / 2
    evaluator.drop_resolved(1)

    remaining = triggers_for(evaluator, "warrior_1")
    assert Trigger.T4_OWN_STATE_THRESHOLD not in remaining


def test_several_triggers_accumulate_onto_one_request(rig) -> None:
    config, engine, messages, evaluator = rig
    messages.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.REQUEST,
        tick=1,
    )
    engine.state.account("warrior_1").gold = config.triggers.t4_gold_low - 1
    evaluator.evaluate(config.triggers.t5_inactivity_ticks)

    assert len(evaluator.pending) >= 1
    assert len(triggers_for(evaluator, "warrior_1")) >= 3


def test_every_trigger_firing_is_logged(rig) -> None:
    """Design v1.0 s15 requires T1-T5 logged per agent and tick."""
    _, engine, messages, evaluator = rig
    messages.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.REQUEST,
        tick=1,
    )
    evaluator.evaluate(1)
    engine.log.close()

    fired = [e for e in read_events(engine.log.path) if e.type is EventType.TRIGGER_FIRED]
    assert any(
        e.agent_id == "warrior_1" and e.payload["trigger"] == Trigger.T1_MESSAGE_RECEIVED.value
        for e in fired
    )
    requested = [
        e for e in read_events(engine.log.path) if e.type is EventType.ACTIVATION_REQUESTED
    ]
    assert any(e.agent_id == "warrior_1" for e in requested)


def test_a_repeat_firing_is_logged_as_pending_and_leaves_the_request_alone(rig) -> None:
    """Every firing per agent and tick (design s15), the request unchanged."""
    _, engine, messages, evaluator = rig
    messages.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.REQUEST,
        tick=1,
    )
    evaluator.evaluate(1)
    request = evaluator.pending["warrior_1"]
    before = (request.first_requested_tick, set(request.triggers))
    evaluator.evaluate(2)  # unserved: T1 still holds and fires again
    after = evaluator.pending["warrior_1"]
    engine.log.close()

    assert after is request
    assert (after.first_requested_tick, set(after.triggers)) == before
    t1 = [
        (e.tick, e.payload["pending"])
        for e in read_events(engine.log.path)
        if e.type is EventType.TRIGGER_FIRED
        and e.agent_id == "warrior_1"
        and e.payload["trigger"] == Trigger.T1_MESSAGE_RECEIVED.value
    ]
    assert [pending for _, pending in t1] == [False, True]
    # The request is announced once, when it is first generated.
    requested = [
        e
        for e in read_events(engine.log.path)
        if e.type is EventType.ACTIVATION_REQUESTED and e.agent_id == "warrior_1"
    ]
    assert len(requested) == 1


def test_all_five_triggers_exist_and_are_distinct(rig) -> None:
    assert len(Trigger) == 5
    assert len({t.value for t in Trigger}) == 5


def test_a_request_dropped_as_resolved_is_logged(rig) -> None:
    """A request that vanishes unserved is accounted for in the log (design s15)."""

    config, engine, _, evaluator = rig
    account = engine.state.account("warrior_1")
    account.gold = config.triggers.t4_gold_low - 1
    evaluator.evaluate(1)
    assert triggers_for(evaluator, "warrior_1") == {Trigger.T4_OWN_STATE_THRESHOLD}

    account.gold = (config.triggers.t4_gold_low + config.triggers.t4_gold_high) / 2
    evaluator.drop_resolved(1)
    assert "warrior_1" not in evaluator.pending
    engine.log.close()

    resolved = [
        e for e in read_events(engine.log.path) if e.type is EventType.ACTIVATION_REQUEST_RESOLVED
    ]
    assert [(e.agent_id, e.payload["triggers"]) for e in resolved] == [
        ("warrior_1", [Trigger.T4_OWN_STATE_THRESHOLD.value])
    ]
    assert resolved[0].payload["waiting_since_tick"] == 1
