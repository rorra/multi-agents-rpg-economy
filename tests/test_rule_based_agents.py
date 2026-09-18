"""Rule-based agents and the action schema (design v1.0 s7)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.agents import (
    AgentAction,
    RuleBasedDecisionPhase,
    execute,
    policy_for,
    reference_price,
)
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import (
    ActionType,
    AdventurerSubclass,
    MessageType,
    OrderSide,
    Priority,
    Product,
    Resource,
    Role,
)
from rpg_economy.engine import Engine, build_world
from rpg_economy.logging_ import EventLogWriter, EventType, read_events
from tests.conftest import make_metadata

IRON = Resource.IRON

#: The potions some subclass takes on an expedition.
PROVISIONS = (Product.HEALTH_POTION, Product.MANA_POTION)


def orders_of(action: AgentAction) -> list[AgentAction]:
    """The market orders an action places: its orders if place_orders, else itself."""
    return list(action.orders) or [action]


def fully_equip(account) -> None:
    """Equip every durable item of the account's gear, so gear needs are met."""
    from tests.conftest import make_config

    config = make_config()
    gear = config.expeditions.gear[AdventurerSubclass(account.specialization)]
    for product in gear:
        if product in config.economy.durability:
            account.equipment[product] = 30


@pytest.fixture
def live_engine(tmp_path: Path, config: SimulationConfig):
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        yield Engine(config, state, log)


# ---------------------------------------------------------------------------
# The action schema
# ---------------------------------------------------------------------------
def test_action_carries_every_field_the_design_names() -> None:
    """`{action, target, quantity, price, recipient, priority, short_reason, message_type}`."""
    action = AgentAction(
        action=ActionType.SELL,
        target="iron",
        quantity=3,
        price=Decimal("10.50"),
        recipient=None,
        priority=Priority.HIGH,
        short_reason="surplus",
    )
    payload = action.to_payload()
    assert set(payload) == {
        "action",
        "target",
        "quantity",
        "price",
        "recipient",
        "priority",
        "short_reason",
        "message_type",
        "orders",
    }
    assert payload["price"] == "10.50"
    assert payload["orders"] is None


def test_director_payload_strips_short_reason() -> None:
    """Design s3.1.2: no free text reaches the Director."""
    action = AgentAction(action=ActionType.WAIT, short_reason="private reasoning")
    assert "short_reason" not in action.to_director_payload()
    assert "private reasoning" not in str(action.to_director_payload())


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------
def test_executor_places_an_order(live_engine: Engine) -> None:
    live_engine.state.account("woodworking_1").receive(IRON, 5)
    outcome = execute(
        live_engine,
        "woodworking_1",
        AgentAction(action=ActionType.SELL, target="iron", quantity=2, price=Decimal("10")),
    )
    assert outcome.accepted and outcome.engine_operation
    assert live_engine.market.best_ask(IRON) == Decimal("10")


def test_wait_is_accepted_but_is_not_an_engine_operation(live_engine: Engine) -> None:
    """Feeds the wasted-activation predicate of design s11.2."""
    outcome = execute(live_engine, "woodworking_1", AgentAction(action=ActionType.WAIT))
    assert outcome.accepted
    assert not outcome.engine_operation


def test_an_invalid_action_is_rejected_without_half_executing(live_engine: Engine) -> None:
    gold_before = live_engine.state.account("woodworking_1").gold
    outcome = execute(
        live_engine,
        "woodworking_1",
        AgentAction(action=ActionType.SELL, target="iron", quantity=99, price=Decimal("10")),
    )
    assert not outcome.accepted
    assert live_engine.state.account("woodworking_1").gold == gold_before
    assert live_engine.market.best_ask(IRON) is None


def test_rejection_is_logged(tmp_path: Path, config: SimulationConfig) -> None:
    path = tmp_path / "run.jsonl"
    state = build_world(config)
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        execute(
            engine,
            "woodworking_1",
            AgentAction(action=ActionType.PRODUCE, target="melee_weapon", quantity=1),
        )
    types = [e.type for e in read_events(path)]
    assert EventType.ACTION_REJECTED in types
    assert EventType.ACTION_EXECUTED not in types


def test_every_action_attempt_is_logged(tmp_path: Path, config: SimulationConfig) -> None:
    path = tmp_path / "run.jsonl"
    state = build_world(config)
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        execute(engine, "woodworking_1", AgentAction(action=ActionType.WAIT, short_reason="idle"))

    outputs = [e for e in read_events(path) if e.type is EventType.AGENT_OUTPUT]
    assert len(outputs) == 1
    assert outputs[0].payload["short_reason"] == "idle"


def test_messaging_is_not_available_before_phase_3(live_engine: Engine) -> None:
    outcome = execute(
        live_engine,
        "woodworking_1",
        AgentAction(action=ActionType.SEND_MESSAGE, recipient="woodworking_2"),
    )
    assert not outcome.accepted


# ---------------------------------------------------------------------------
# Reference price
# ---------------------------------------------------------------------------
def test_reference_price_falls_back_to_configuration(live_engine: Engine) -> None:
    """There is no official price, so an untraded good has only its config value."""
    assert reference_price(live_engine, IRON) == Decimal("10.00")


def test_reference_price_follows_recent_trades(live_engine: Engine) -> None:
    live_engine.state.account("woodworking_1").receive(IRON, 5)
    live_engine.submit_order("woodworking_1", OrderSide.SELL, IRON, 2, "20")
    live_engine.submit_order("woodworking_2", OrderSide.BUY, IRON, 2, "20")

    from rpg_economy.agents.rule_based import REFERENCE_ANCHOR

    start = Decimal(live_engine.config.economy.initial_reference_price["iron"])
    expected = (1 - REFERENCE_ANCHOR) * Decimal(20) + REFERENCE_ANCHOR * start
    assert reference_price(live_engine, IRON) == expected.quantize(Decimal("0.01"))


def test_an_untraded_good_is_believed_worth_its_starting_price(live_engine: Engine) -> None:
    start = Decimal(live_engine.config.economy.initial_reference_price["iron"])
    assert reference_price(live_engine, IRON) == start.quantize(Decimal("0.01"))


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def test_an_away_adventurer_waits(live_engine: Engine, config: SimulationConfig) -> None:
    account = live_engine.state.account("warrior_1")
    live_engine.start_expedition("warrior_1", "green_hollow")

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("warrior_1")
    )
    assert action.action is ActionType.WAIT


def test_an_adventurer_equips_owned_gear_before_departing(
    live_engine: Engine, config: SimulationConfig
) -> None:
    account = live_engine.state.account("warrior_1")
    account.equipment.clear()  # the starting weapon broke
    account.receive(Product.MELEE_WEAPON, 1)

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("warrior_1")
    )
    assert action.action is ActionType.EQUIP
    assert action.target == Product.MELEE_WEAPON.value


def test_an_idle_adventurer_goes_on_an_expedition(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """Equipped, stocked, in town and with nothing to sell: go gather."""
    account = live_engine.state.account("ranger_1")
    fully_equip(account)
    for product in PROVISIONS:
        account.receive(product, 5)

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("ranger_1")
    )
    assert action.action is ActionType.START_EXPEDITION
    assert action.target in live_engine.zones.zone_ids


def test_an_unequipped_adventurer_buys_its_main_gear_first(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """The most valuable item first: it adds the most to the chance of success."""
    account = live_engine.state.account("ranger_1")
    account.equipment.clear()  # the starting bow broke

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("ranger_1")
    )
    first = orders_of(action)[0]
    assert first.action is ActionType.BUY
    assert first.target == Product.BOW_GEAR.value
    assert first.priority is Priority.HIGH


def test_an_adventurer_without_gear_still_sets_out(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """Gear is optional: with nothing to trade and nothing to buy with, it sets out."""
    account = live_engine.state.account("warrior_1")
    account.equipment.clear()
    account.inventory.clear()
    account.gold = Decimal("0")
    policy = policy_for(Role.ADVENTURER, config)
    actions = [
        policy.decide(live_engine, account, live_engine.random.agent("warrior_1")).action
        for _ in range(3)
    ]
    assert ActionType.START_EXPEDITION in actions


def test_an_adventurer_sells_its_surplus(live_engine: Engine, config: SimulationConfig) -> None:
    """Selling is housekeeping: it happens once real needs are covered."""
    account = live_engine.state.account("warrior_1")
    fully_equip(account)
    for product in PROVISIONS:
        account.receive(product, 5)
    account.receive(IRON, 20)

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("warrior_1")
    )
    assert action.action is ActionType.SELL
    assert action.target == "iron"


def test_a_stocked_artisan_produces(live_engine: Engine, config: SimulationConfig) -> None:
    account = live_engine.state.account("smithing_1")
    action = policy_for(Role.ARTISAN, config).decide(
        live_engine, account, live_engine.random.agent("smithing_1")
    )
    assert action.action is ActionType.PRODUCE


def test_an_artisan_without_inputs_buys_them(live_engine: Engine, config: SimulationConfig) -> None:
    account = live_engine.state.account("smithing_1")
    account.inventory.clear()

    action = policy_for(Role.ARTISAN, config).decide(
        live_engine, account, live_engine.random.agent("smithing_1")
    )
    buys = [o for o in orders_of(action) if o.action is ActionType.BUY]
    assert buys and all(o.priority is Priority.HIGH for o in buys)
    # Every input the line lacks, in one action.
    assert {o.target for o in buys} == {"iron", "leather"}


def test_an_artisan_at_capacity_sells_finished_stock(
    live_engine: Engine, config: SimulationConfig
) -> None:
    account = live_engine.state.account("smithing_1")
    account.capacity_credit = Decimal(0)
    account.receive(Product.MELEE_WEAPON, 3)

    action = policy_for(Role.ARTISAN, config).decide(
        live_engine, account, live_engine.random.agent("smithing_1")
    )
    assert action.action is ActionType.SELL
    assert action.target == Product.MELEE_WEAPON.value


def _two_sided_books(engine: Engine) -> None:
    """A resting bid and ask on every resource, well apart so neither crosses."""
    engine.state.account("smithing_1").gold = Decimal(1000)
    for resource in Resource:
        reference = reference_price(engine, resource)
        engine.state.account("warrior_1").receive(resource, 1)
        engine.submit_order("warrior_1", OrderSide.SELL, resource, 1, str(reference * 2))
        engine.submit_order("smithing_1", OrderSide.BUY, resource, 1, str(reference / 2))


def test_policies_exist_for_every_role(config: SimulationConfig) -> None:
    for role in Role:
        assert policy_for(role, config) is not None


# ---------------------------------------------------------------------------
# The decision phase
# ---------------------------------------------------------------------------
def test_every_agent_is_activated_once_per_cycle(tmp_path: Path, config: SimulationConfig) -> None:
    path = tmp_path / "run.jsonl"
    state = build_world(config)
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, state, log)
        engine.run_cycle(RuleBasedDecisionPhase(config))

    outputs = [e for e in read_events(path) if e.type is EventType.AGENT_OUTPUT]
    assert [e.agent_id for e in outputs] == state.agent_ids


def test_activations_see_the_previous_activations_effects(
    tmp_path: Path, config: SimulationConfig
) -> None:
    """The sequencing rule design s8.1 fixes for all three conditions."""
    path = tmp_path / "run.jsonl"
    state = build_world(config)
    observed: list[int] = []

    def phase(engine: Engine) -> None:
        engine.state.account("woodworking_1").receive(IRON, 5)
        engine.submit_order("woodworking_1", OrderSide.SELL, IRON, 1, "10")
        observed.append(len(engine.market.resting_orders(IRON, OrderSide.SELL)))
        engine.submit_order("woodworking_1", OrderSide.SELL, IRON, 1, "11")
        observed.append(len(engine.market.resting_orders(IRON, OrderSide.SELL)))

    with EventLogWriter(path, make_metadata(config)) as log:
        Engine(config, state, log).run_cycle(phase)

    assert observed == [1, 2]


def test_needs_come_before_housekeeping(live_engine: Engine, config: SimulationConfig) -> None:
    """An adventurer with a big haul must still re-equip.

    Selling first would lock a stocked adventurer into selling forever: its
    inventory never falls below the sell threshold, so it never reaches the
    branch that replaces broken equipment.
    """
    account = live_engine.state.account("warrior_1")
    account.receive(IRON, 40)
    account.equipment.clear()  # the starting weapon broke

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("warrior_1")
    )
    first = orders_of(action)[0]
    assert first.action is ActionType.BUY
    assert first.target == Product.MELEE_WEAPON.value
    # ... and the haul goes up for sale in the same action.
    assert any(o.action is ActionType.SELL and o.target == "iron" for o in orders_of(action))


def test_an_adventurer_restocks_consumables(live_engine: Engine, config: SimulationConfig) -> None:
    """Without this nothing ever buys a potion and the alchemist has no market."""
    account = live_engine.state.account("warrior_1")
    fully_equip(account)
    account.inventory.clear()

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("warrior_1")
    )
    assert action.action is ActionType.BUY
    assert action.target in {p.value for p in PROVISIONS}


def test_a_pending_bid_for_one_potion_does_not_block_restocking_the_other(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """A mage waiting on a health potion bid still restocks mana."""
    account = live_engine.state.account("mage_1")
    gear = config.expeditions.gear[AdventurerSubclass.MAGE]
    assert set(PROVISIONS) <= set(gear), "a mage takes both potions"
    fully_equip(account)
    account.inventory.clear()
    account.gold = Decimal("500")
    live_engine.submit_order("mage_1", OrderSide.BUY, Product.HEALTH_POTION, 1, Decimal("1"))

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("mage_1")
    )
    bought = [o.target for o in orders_of(action) if o.action is ActionType.BUY]
    assert Product.MANA_POTION.value in bought
    assert Product.HEALTH_POTION.value not in bought, "one resting bid per good"


def test_an_adventurer_short_of_both_potions_restocks_both(
    live_engine: Engine, config: SimulationConfig
) -> None:
    account = live_engine.state.account("mage_1")
    fully_equip(account)
    account.inventory.clear()
    account.gold = Decimal("500")

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("mage_1")
    )
    bought = {o.target for o in orders_of(action) if o.action is ActionType.BUY}
    assert bought == {p.value for p in PROVISIONS}


def test_an_artisan_does_not_stockpile_unsold_output(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """Overproduction converts saleable inputs into stock nobody wants."""
    from rpg_economy.agents.rule_based import ArtisanPolicy

    account = live_engine.state.account("smithing_1")
    for resource in Resource:
        account.receive(resource, 40)
    account.receive(Product.MELEE_WEAPON, ArtisanPolicy.MAX_FINISHED_STOCK)
    account.receive(Product.ARMOR, ArtisanPolicy.MAX_FINISHED_STOCK)

    action = policy_for(Role.ARTISAN, config).decide(
        live_engine, account, live_engine.random.agent("smithing_1")
    )
    assert action.action is not ActionType.PRODUCE


# ---------------------------------------------------------------------------
# Actions the engine must reject rather than crash on
# ---------------------------------------------------------------------------
def _executing_engine(tmp_path: Path):
    from tests.conftest import make_config

    config = make_config(run={"condition": "B"})
    log = EventLogWriter(tmp_path / "run.jsonl", make_metadata(config))
    return Engine(config, build_world(config), log), log


@pytest.mark.invariant
def test_a_foreign_cancel_is_a_rejected_action(tmp_path: Path) -> None:
    from rpg_economy.agents import AgentAction, execute

    engine, log = _executing_engine(tmp_path)
    with log:
        engine.state.account("smithing_1").receive(Resource.IRON, 1)
        order = engine.submit_order("smithing_1", OrderSide.SELL, Resource.IRON, 1, Decimal("50"))
        outcome = execute(
            engine,
            "woodworking_1",
            AgentAction(action=ActionType.CANCEL_ORDER, quantity=order.order_id),
        )
    assert not outcome.accepted
    assert order.is_resting


@pytest.mark.invariant
@pytest.mark.parametrize("price", ["1e26", "1e30"])
def test_an_absurd_price_is_a_rejected_order_not_a_crash(tmp_path: Path, price: str) -> None:
    """A price too large to quantize is a malformed order, not a bad quantity."""
    from rpg_economy.agents import AgentAction, execute
    from rpg_economy.domain import RejectionCode

    engine, log = _executing_engine(tmp_path)
    buy = AgentAction(action=ActionType.BUY, target="iron", quantity=1, price=Decimal(price))
    with log:
        outcome = execute(engine, "woodworking_1", buy)
        batch = execute(
            engine,
            "woodworking_1",
            AgentAction(action=ActionType.PLACE_ORDERS, orders=(buy,)),
        )
    assert not outcome.accepted
    assert outcome.code is RejectionCode.INVALID_ORDER
    assert "too large" in outcome.detail
    assert not batch.accepted
    assert batch.code is RejectionCode.INVALID_ORDER
    assert batch.order_results[0]["rejection"] == RejectionCode.INVALID_ORDER.value


@pytest.mark.invariant
@pytest.mark.parametrize("stray", [ActionType.START_EXPEDITION, ActionType.PRODUCE])
def test_place_orders_carrying_a_non_order_is_rejected_whole(
    tmp_path: Path, stray: ActionType
) -> None:
    """The executor re-checks the batch shape parsing enforces, before any order books."""
    from rpg_economy.agents import AgentAction, execute
    from rpg_economy.domain import RejectionCode

    engine, log = _executing_engine(tmp_path)
    sell = AgentAction(action=ActionType.SELL, target="iron", quantity=1, price=Decimal("50"))
    with log:
        engine.state.account("smithing_1").receive(Resource.IRON, 1)
        outcome = execute(
            engine,
            "smithing_1",
            AgentAction(
                action=ActionType.PLACE_ORDERS,
                orders=(sell, AgentAction(action=stray, target="melee_weapon", quantity=1)),
            ),
        )
    assert not outcome.accepted
    assert not outcome.engine_operation
    assert outcome.code is RejectionCode.INVALID_ORDER
    assert not engine.market.open_orders_of("smithing_1")


@pytest.mark.invariant
def test_place_orders_over_the_cap_is_rejected_whole(tmp_path: Path) -> None:
    from rpg_economy.agents import AgentAction, execute
    from rpg_economy.domain import RejectionCode

    engine, log = _executing_engine(tmp_path)
    cap = engine.config.market.max_orders_per_activation
    buy = AgentAction(action=ActionType.BUY, target="iron", quantity=1, price=Decimal("1"))
    with log:
        outcome = execute(
            engine,
            "woodworking_1",
            AgentAction(action=ActionType.PLACE_ORDERS, orders=(buy,) * (cap + 1)),
        )
        at_cap = execute(
            engine,
            "woodworking_1",
            AgentAction(action=ActionType.PLACE_ORDERS, orders=(buy,) * cap),
        )
    assert not outcome.accepted
    assert outcome.code is RejectionCode.INVALID_ORDER
    assert at_cap.accepted
    assert len(engine.market.open_orders_of("woodworking_1")) == cap


@pytest.mark.invariant
def test_a_broken_state_invariant_propagates_instead_of_being_a_rejection(
    tmp_path: Path,
) -> None:
    """A mid-trade invariant failure ends the run; it is never logged as a refusal."""
    from rpg_economy.agents import AgentAction, execute
    from rpg_economy.engine import EngineInvariantError

    engine, log = _executing_engine(tmp_path)
    with log:
        buyer = engine.state.account("woodworking_1")
        buyer.gold = Decimal("100")
        engine.submit_order("woodworking_1", OrderSide.BUY, Resource.IRON, 1, Decimal("10"))
        buyer.reserved_gold = Decimal(0)  # corrupt the escrow behind the resting bid
        engine.state.account("smithing_1").receive(Resource.IRON, 1)
        with pytest.raises(EngineInvariantError):
            execute(
                engine,
                "smithing_1",
                AgentAction(action=ActionType.SELL, target="iron", quantity=1, price=Decimal("10")),
            )
    events = read_events(tmp_path / "run.jsonl")
    assert not any(e.type is EventType.ACTION_REJECTED for e in events)


def test_the_activation_key_is_added_to_every_executor_event(tmp_path: Path) -> None:
    from rpg_economy.agents import AgentAction, execute

    key = {"tick": 0, "agent_id": "smithing_1", "seq": 3}
    engine, log = _executing_engine(tmp_path)
    sell = AgentAction(action=ActionType.SELL, target="iron", quantity=1, price=Decimal("50"))
    with log:
        engine.state.account("smithing_1").inventory = {Resource.IRON: 1}
        execute(engine, "smithing_1", sell, activation=key)
        execute(engine, "smithing_1", sell, activation=key)  # nothing left to sell
        execute(
            engine,
            "smithing_1",
            AgentAction(action=ActionType.PLACE_ORDERS, orders=(sell,)),
            activation=key,
        )
        execute(engine, "smithing_1", AgentAction(action=ActionType.WAIT))
    events = read_events(tmp_path / "run.jsonl")
    kinds = {EventType.AGENT_OUTPUT, EventType.ACTION_EXECUTED, EventType.ACTION_REJECTED}
    logged = [e for e in events if e.type in kinds]
    keyed = [e for e in logged if "activation" in e.payload]
    assert {e.type for e in keyed} == kinds
    assert all(e.payload["activation"] == key for e in keyed)
    # Without a key, the payloads are as before.
    unkeyed = [e for e in logged if "activation" not in e.payload]
    assert [e.type for e in unkeyed] == [EventType.AGENT_OUTPUT, EventType.ACTION_EXECUTED]


# ---------------------------------------------------------------------------
# Offline agents stay solvent enough to trade (the calibration proxy)
# ---------------------------------------------------------------------------
def test_an_artisan_does_not_stack_bids_for_one_input(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """A second bid for the same input would lock its gold and leave it idle."""
    from rpg_economy.agents.executor import execute

    account = live_engine.state.account("alchemy_1")
    account.inventory.clear()
    policy = policy_for(Role.ARTISAN, config)
    bought: list[str] = []
    for _ in range(4):
        action = policy.decide(live_engine, account, live_engine.random.agent("alchemy_1"))
        if action.action is ActionType.BUY:
            bought.append(action.target)
            execute(live_engine, "alchemy_1", action)
    assert len(bought) == len(set(bought)), f"bid twice for the same input: {bought}"


def test_an_artisan_sources_the_line_buyers_are_bidding_for(
    live_engine: Engine, config: SimulationConfig
) -> None:
    from rpg_economy.agents.executor import execute

    buyer = live_engine.state.account("warrior_1")
    buyer.gold = Decimal("500")
    execute(
        live_engine,
        "warrior_1",
        AgentAction(
            action=ActionType.BUY,
            target=Product.HEALTH_POTION.value,
            quantity=1,
            price=Decimal("40"),
        ),
    )
    account = live_engine.state.account("alchemy_1")
    account.inventory.clear()
    action = policy_for(Role.ARTISAN, config).decide(
        live_engine, account, live_engine.random.agent("alchemy_1")
    )
    first = orders_of(action)[0]
    assert first.action is ActionType.BUY
    assert first.target == Resource.HERBS.value


# ---------------------------------------------------------------------------
# Two-sided price formation in the offline agents
# ---------------------------------------------------------------------------
def test_an_adventurer_sells_into_a_bid_at_the_recent_price(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """A buyer waiting at the going price gets the iron, however little is held."""
    from rpg_economy.agents.executor import execute
    from rpg_economy.agents.rule_based import reference_price

    smith = live_engine.state.account("smithing_1")
    smith.gold = Decimal("500")
    price = reference_price(live_engine, IRON)
    execute(
        live_engine,
        "smithing_1",
        AgentAction(action=ActionType.BUY, target="iron", quantity=2, price=price),
    )
    account = live_engine.state.account("warrior_1")
    fully_equip(account)
    for product in PROVISIONS:
        account.receive(product, 5)
    account.receive(IRON, 1)  # below the surplus threshold

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("warrior_1")
    )
    assert action.action is ActionType.SELL
    assert action.target == "iron"
    assert action.price == price


def test_a_seller_concedes_while_unsold_and_asks_more_while_selling(
    live_engine: Engine, config: SimulationConfig
) -> None:
    from rpg_economy.agents.rule_based import MAX_SELLER_DISCOUNT, MAX_SELLER_PREMIUM

    policy = policy_for(Role.ADVENTURER, config)
    account = live_engine.state.account("warrior_1")
    account.receive(IRON, 10)
    shading = [policy._unsold(account, IRON) for _ in range(4)]
    assert shading[0] == 0
    assert shading[1] < shading[2] < shading[3] <= MAX_SELLER_DISCOUNT

    # Each offer that finds its stock sold since asks a step more.
    premiums = []
    for _ in range(3):
        account.consume(IRON, 1)
        premiums.append(policy._unsold(account, IRON))
    assert 0 > premiums[0] > premiums[1] > premiums[2] >= -MAX_SELLER_PREMIUM


def test_escalation_resets_once_the_buyer_is_filled(config: SimulationConfig) -> None:
    from rpg_economy.agents.rule_based import AdventurerPolicy
    from rpg_economy.engine import build_world

    account = build_world(config).account("warrior_1")
    policy = AdventurerPolicy(config)
    first = policy._escalate("warrior_1", Product.HEALTH_POTION, account)
    second = policy._escalate("warrior_1", Product.HEALTH_POTION, account)
    assert second > first
    account.receive(Product.HEALTH_POTION, 1)
    assert policy._escalate("warrior_1", Product.HEALTH_POTION, account) == first


def test_a_balanced_market_does_not_drift(config: SimulationConfig) -> None:
    """Neither side quotes off the recent price unless pressed (no ratchet)."""
    from rpg_economy.agents.rule_based import BUYER_PREMIUM, SELLER_MARGIN, RuleBasedPolicy

    assert SELLER_MARGIN == 0
    assert -RuleBasedPolicy.ESCALATION_STEP < BUYER_PREMIUM <= 0


# ---------------------------------------------------------------------------
# A refusal carries a closed code beside the engine's words
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("agent_id", "action", "code"),
    [
        (
            "woodworking_1",
            AgentAction(action=ActionType.SELL, target="iron", quantity=99, price=Decimal("10")),
            "insufficient_inventory",
        ),
        (
            "woodworking_1",
            AgentAction(action=ActionType.BUY, target="iron", quantity=1, price=Decimal("99999")),
            "insufficient_gold",
        ),
        (
            "woodworking_1",
            AgentAction(action=ActionType.PRODUCE, target="melee_weapon", quantity=1),
            "not_allowed",
        ),
        (
            "smithing_1",
            AgentAction(action=ActionType.PRODUCE, target="health_potion", quantity=1),
            "not_allowed",
        ),
        (
            "smithing_1",
            AgentAction(action=ActionType.PRODUCE, target="melee_weapon", quantity=1000),
            "no_capacity",
        ),
        (
            "woodworking_1",
            AgentAction(action=ActionType.CANCEL_ORDER, quantity=999),
            "invalid_order",
        ),
        (
            "warrior_1",
            AgentAction(action=ActionType.START_EXPEDITION, target="the_moon"),
            "invalid_target",
        ),
        (
            "woodworking_1",
            AgentAction(action=ActionType.START_EXPEDITION, target="green_hollow"),
            "not_allowed",
        ),
        (
            "warrior_1",
            AgentAction(action=ActionType.EQUIP, target="melee_weapon"),
            "insufficient_inventory",
        ),
        (
            "woodworking_1",
            AgentAction(action=ActionType.SELL, target="gold", quantity=1, price=Decimal("1")),
            "invalid_target",
        ),
        (
            "woodworking_1",
            AgentAction(action=ActionType.BUY, target="iron", quantity=1, price=Decimal("0")),
            "invalid_order",
        ),
        (
            "woodworking_1",
            AgentAction(action=ActionType.BUY, target="iron", quantity=0, price=Decimal("1")),
            "invalid_quantity",
        ),
        (
            "woodworking_1",
            AgentAction(action=ActionType.EQUIP, target="melee_weapon"),
            "not_allowed",
        ),
    ],
)
def test_a_refusal_is_classified_into_a_closed_code(
    tmp_path: Path, config: SimulationConfig, agent_id, action, code
) -> None:
    from rpg_economy.domain import RejectionCode

    path = tmp_path / "run.jsonl"
    with EventLogWriter(path, make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        outcome = execute(engine, agent_id, action)

    assert not outcome.accepted
    assert outcome.code is RejectionCode(code)
    assert outcome.detail
    rejected = next(e for e in read_events(path) if e.type is EventType.ACTION_REJECTED)
    assert rejected.payload["code"] == code
    assert rejected.payload["reason"] == outcome.detail


def test_missing_inputs_have_their_own_code(live_engine: Engine) -> None:
    from rpg_economy.domain import RejectionCode

    live_engine.state.account("smithing_1").inventory.clear()
    outcome = execute(
        live_engine,
        "smithing_1",
        AgentAction(action=ActionType.PRODUCE, target="melee_weapon", quantity=1),
    )
    assert outcome.code is RejectionCode.MISSING_INPUTS


def test_a_departure_while_away_is_its_own_code(live_engine: Engine) -> None:
    from rpg_economy.domain import RejectionCode

    live_engine.start_expedition("warrior_1", "green_hollow")
    outcome = execute(
        live_engine,
        "warrior_1",
        AgentAction(action=ActionType.START_EXPEDITION, target="green_hollow"),
    )
    assert outcome.code is RejectionCode.AGENT_AWAY


def test_a_departure_while_resting_is_its_own_code(live_engine: Engine) -> None:
    """Resting in town is not away: the RESULT tells the two apart, as the Director's legend does."""
    from rpg_economy.domain import RejectionCode

    account = live_engine.state.account("warrior_1")
    account.recovering_until_tick = live_engine.tick + 3
    outcome = execute(
        live_engine,
        "warrior_1",
        AgentAction(action=ActionType.START_EXPEDITION, target="green_hollow"),
    )
    assert outcome.code is RejectionCode.RESTING
    assert "resting" in outcome.detail
    assert account.expedition is None


def test_equipping_while_on_an_expedition_is_refused_as_away(live_engine: Engine) -> None:
    """Gear is fixed when an adventurer sets out; trading while away stays allowed."""
    from rpg_economy.domain import RejectionCode

    account = live_engine.state.account("warrior_1")
    live_engine.start_expedition("warrior_1", "green_hollow")
    account.receive(Product.MELEE_WEAPON, 1)
    before = dict(account.equipment)

    outcome = execute(
        live_engine, "warrior_1", AgentAction(action=ActionType.EQUIP, target="melee_weapon")
    )
    assert outcome.code is RejectionCode.AGENT_AWAY
    assert account.equipment == before
    assert account.quantity(Product.MELEE_WEAPON) == 1

    account.receive(Resource.IRON, 1)
    sell = AgentAction(action=ActionType.SELL, target="iron", quantity=1, price=Decimal("10"))
    assert execute(live_engine, "warrior_1", sell).accepted


def test_an_artisan_cannot_equip_even_what_it_holds(live_engine: Engine) -> None:
    from rpg_economy.domain import RejectionCode

    account = live_engine.state.account("smithing_1")
    account.receive(Product.MELEE_WEAPON, 1)
    outcome = execute(
        live_engine, "smithing_1", AgentAction(action=ActionType.EQUIP, target="melee_weapon")
    )
    assert outcome.code is RejectionCode.NOT_ALLOWED
    assert account.equipment == {}
    assert account.quantity(Product.MELEE_WEAPON) >= 1


def test_equipping_a_non_equipment_product_is_an_invalid_target(live_engine: Engine) -> None:
    """Not equipment is checked before stock: holding none must not read as a shortage."""
    from rpg_economy.domain import RejectionCode

    account = live_engine.state.account("warrior_1")
    account.inventory.pop(Product.HEALTH_POTION, None)
    outcome = execute(
        live_engine, "warrior_1", AgentAction(action=ActionType.EQUIP, target="health_potion")
    )
    assert outcome.code is RejectionCode.INVALID_TARGET

    account.receive(Product.HEALTH_POTION, 1)
    outcome = execute(
        live_engine, "warrior_1", AgentAction(action=ActionType.EQUIP, target="health_potion")
    )
    assert outcome.code is RejectionCode.INVALID_TARGET
    assert account.quantity(Product.HEALTH_POTION) == 1


def test_equipping_missing_equipment_is_insufficient_inventory(live_engine: Engine) -> None:
    from rpg_economy.domain import RejectionCode

    account = live_engine.state.account("warrior_1")
    account.inventory.pop(Product.ARMOR, None)
    outcome = execute(
        live_engine, "warrior_1", AgentAction(action=ActionType.EQUIP, target="armor")
    )
    assert outcome.code is RejectionCode.INSUFFICIENT_INVENTORY


def test_an_accepted_action_has_no_rejection_code(live_engine: Engine) -> None:
    outcome = execute(live_engine, "woodworking_1", AgentAction(action=ActionType.WAIT))
    assert outcome.accepted
    assert outcome.code is None


def test_an_unrecognised_refusal_falls_back_to_other() -> None:
    from rpg_economy.agents.executor import rejection_code
    from rpg_economy.domain import RejectionCode

    assert rejection_code(RuntimeError("something unforeseen")) is RejectionCode.OTHER
    from rpg_economy.messaging import MessageRejected, NotAReply

    assert rejection_code(NotAReply("x")) is RejectionCode.NOT_A_REPLY
    assert rejection_code(MessageRejected("x")) is RejectionCode.MESSAGE_REJECTED
    assert rejection_code(ZeroDivisionError("x")) is RejectionCode.INVALID_QUANTITY


# ---------------------------------------------------------------------------
# Offline adventurers and artisans keep the market moving
# ---------------------------------------------------------------------------
def test_an_adventurer_offers_even_a_single_gathered_unit(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """With no buyer waiting, anything gathered still goes on the book as an ask."""
    from rpg_economy.agents.rule_based import ADVENTURER_STOCK_THRESHOLD

    assert ADVENTURER_STOCK_THRESHOLD == 0
    account = live_engine.state.account("warrior_1")
    fully_equip(account)
    for product in PROVISIONS:
        account.receive(product, 5)
    account.receive(IRON, 1)
    assert live_engine.market.best_bid(IRON) is None

    action = policy_for(Role.ADVENTURER, config).decide(
        live_engine, account, live_engine.random.agent("warrior_1")
    )
    assert action.action is ActionType.SELL
    assert action.target == "iron"
    assert action.quantity == 1
    execute(live_engine, "warrior_1", action)
    assert live_engine.market.best_ask(IRON) == action.price


def test_an_alchemist_never_brews_past_its_stock_cap(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """The batch counts recipe runs; the cap counts units: four, or the engine's
    craft limit when lower (held at it, the policy kept choosing to craft and
    was refused every time, never reaching its selling branch)."""
    from rpg_economy.agents.rule_based import ArtisanPolicy

    policy = ArtisanPolicy(config)
    account = live_engine.state.account("alchemy_1")
    account.inventory.clear()
    for resource in Resource:
        account.receive(resource, 40)
    potion = Product.HEALTH_POTION
    cap = min(4, config.economy.craft_stock_limit or 4)
    assert policy._stock_cap(potion) == cap
    assert live_engine.production.capacity_of(account) > 2, "capacity must not be what binds"

    produced = 0
    for _ in range(6):
        action = policy.decide(live_engine, account, live_engine.random.agent("alchemy_1"))
        if action.action is not ActionType.PRODUCE:
            continue
        assert execute(live_engine, "alchemy_1", action).accepted
        produced += 1
        for product in PROVISIONS:
            assert account.quantity(product) <= policy._stock_cap(product), product
    assert produced
    assert account.quantity(potion) == cap


def _alchemist_with_stock_on_the_book(engine: Engine):
    from rpg_economy.agents.rule_based import ArtisanPolicy

    policy = ArtisanPolicy(engine.config)
    account = engine.state.account("alchemy_1")
    account.inventory.clear()
    for product in (Product.HEALTH_POTION, Product.MANA_POTION):
        cap = policy._stock_cap(product)
        account.receive(product, cap)
        engine.submit_order("alchemy_1", OrderSide.SELL, product, cap, "500")
        assert account.available_quantity(product) == 0
    return account


def test_an_artisan_counts_stock_on_the_book_before_sourcing(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """Units offered and unsold are still unsold: no inputs for more of them."""
    account = _alchemist_with_stock_on_the_book(live_engine)

    action = policy_for(Role.ARTISAN, config).decide(
        live_engine, account, live_engine.random.agent("alchemy_1")
    )
    assert action.action is not ActionType.BUY
    assert action.action is ActionType.WAIT


def test_an_artisan_counts_stock_on_the_book_before_producing(
    live_engine: Engine, config: SimulationConfig
) -> None:
    account = _alchemist_with_stock_on_the_book(live_engine)
    for resource in Resource:
        account.receive(resource, 40)

    action = policy_for(Role.ARTISAN, config).decide(
        live_engine, account, live_engine.random.agent("alchemy_1")
    )
    assert action.action is not ActionType.PRODUCE


def test_an_adventurer_holds_back_a_resource_while_its_price_is_very_low(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """It waits a while for a better price, then sells at what the market pays."""
    from rpg_economy.agents.rule_based import HOLD_FRACTION, HOLD_PATIENCE, AdventurerPolicy

    policy = AdventurerPolicy(config)
    account = live_engine.state.account("warrior_1")
    start = Decimal(config.economy.initial_reference_price["wood"])
    floor = (start * HOLD_FRACTION).quantize(Decimal("0.01"))
    # A glut: wood last traded at a third of its starting price.
    live_engine.state.account("ranger_1").receive(Resource.WOOD, 1)
    live_engine.submit_order("ranger_1", OrderSide.SELL, Resource.WOOD, 1, str(start / 3))
    live_engine.submit_order("smithing_1", OrderSide.BUY, Resource.WOOD, 1, str(start / 3))

    assert policy._hold_floor(live_engine, account, Resource.WOOD) == floor
    live_engine.state.tick += HOLD_PATIENCE
    assert policy._hold_floor(live_engine, account, Resource.WOOD) == Decimal(0)


def test_an_adventurer_sells_freely_at_a_normal_price(
    live_engine: Engine, config: SimulationConfig
) -> None:
    from rpg_economy.agents.rule_based import AdventurerPolicy

    account = live_engine.state.account("warrior_1")
    assert AdventurerPolicy(config)._hold_floor(live_engine, account, IRON) == Decimal(0)


def test_an_artisan_never_bids_more_for_an_input_than_its_product_returns(
    live_engine: Engine, config: SimulationConfig
) -> None:
    from rpg_economy.agents.rule_based import ArtisanPolicy

    policy = ArtisanPolicy(config)
    recipe = config.economy.recipes[Product.MELEE_WEAPON]
    ceiling = policy._input_ceiling(live_engine, Product.MELEE_WEAPON, IRON)
    value = reference_price(live_engine, Product.MELEE_WEAPON) * recipe.output_quantity
    leather = reference_price(live_engine, Resource.LEATHER) * recipe.inputs[Resource.LEATHER]
    assert ceiling == ((value - leather) / recipe.inputs[IRON]).quantize(Decimal("0.01"))

    # Blocked for a long time, the escalated bid still stops at the ceiling.
    account = live_engine.state.account("smithing_1")
    account.inventory.clear()
    for _ in range(40):
        action = policy.decide(live_engine, account, live_engine.random.agent("smithing_1"))
        if action.action is ActionType.BUY and action.target == IRON.value:
            assert action.price <= ceiling


def _offered(tmp_path: Path, price: str = "9", quantity: int = 2):
    """warrior_1 has offered woodworking_1 iron; returns engine, log and bus."""
    from rpg_economy.messaging import MessageBus
    from rpg_economy.orchestration import build_directory

    engine, log = _executing_engine(tmp_path)
    bus = MessageBus(build_directory(engine))
    bus.send(
        sender_id="warrior_1",
        recipient_id="woodworking_1",
        message_type=MessageType.OFFER,
        tick=0,
        payload={"good": "iron", "quantity": quantity, "price": price},
    )
    return engine, log, bus


def _ack(
    quantity: int | None = 2, price: str | None = "9", recipient: str = "warrior_1"
) -> AgentAction:
    return AgentAction(
        action=ActionType.SEND_MESSAGE,
        message_type=MessageType.ACK,
        target="iron",
        quantity=quantity,
        price=None if price is None else Decimal(price),
        recipient=recipient,
    )


def test_accepting_an_offer_places_the_buy_in_the_same_action(tmp_path: Path) -> None:
    engine, log, bus = _offered(tmp_path)
    with log:
        outcome = execute(engine, "woodworking_1", _ack(), messages=bus)
    assert outcome.accepted
    bids = [o for o in engine.market.open_orders_of("woodworking_1") if o.side is OrderSide.BUY]
    assert [(o.good, o.quantity, o.limit_price) for o in bids] == [(IRON, 2, Decimal("9.00"))]
    assert bus.inbox("warrior_1")[-1].type is MessageType.ACK


def test_message_sent_names_the_bus_message_id(tmp_path: Path) -> None:
    """message_sent carries the id message_delivered names, so the two join."""
    engine, log, bus = _offered(tmp_path)
    with log:
        assert execute(engine, "woodworking_1", _ack(), messages=bus).accepted
    sent = [e for e in read_events(tmp_path / "run.jsonl") if e.type is EventType.MESSAGE_SENT]
    assert len(sent) == 1
    ack = bus.inbox("warrior_1")[-1]
    assert sent[0].payload["message_id"] == ack.message_id


def test_accepting_a_request_places_the_sell(tmp_path: Path) -> None:
    from rpg_economy.messaging import MessageBus
    from rpg_economy.orchestration import build_directory

    engine, log = _executing_engine(tmp_path)
    bus = MessageBus(build_directory(engine))
    engine.state.account("woodworking_1").receive(IRON, 3)
    bus.send(
        sender_id="smithing_1",
        recipient_id="woodworking_1",
        message_type=MessageType.REQUEST,
        tick=0,
        payload={"good": "iron", "quantity": 3, "price": "11"},
    )
    ack = _ack(quantity=3, price="11", recipient="smithing_1")
    with log:
        assert execute(engine, "woodworking_1", ack, messages=bus).accepted
    asks = [o for o in engine.market.open_orders_of("woodworking_1") if o.side is OrderSide.SELL]
    assert [(o.quantity, o.limit_price) for o in asks] == [(3, Decimal("11.00"))]


def test_an_ack_that_cannot_be_backed_is_not_sent(tmp_path: Path) -> None:
    engine, log, bus = _offered(tmp_path)
    engine.state.account("woodworking_1").gold = Decimal("1")
    with log:
        outcome = execute(engine, "woodworking_1", _ack(), messages=bus)
    from rpg_economy.domain import RejectionCode

    assert not outcome.accepted
    assert outcome.code is RejectionCode.INSUFFICIENT_GOLD
    assert bus.inbox("warrior_1") == []
    # The proposal is still open: the agent can still answer it.
    assert bus.open_proposal("woodworking_1", "warrior_1", "iron") is MessageType.OFFER


def test_an_ack_needs_the_terms_of_its_order() -> None:
    from rpg_economy.agents.parsing import InvalidAgentOutput, parse_action
    from rpg_economy.domain import Condition

    raw = (
        '{"action": "send_message", "message_type": "ACK", "target": "iron", '
        '"recipient": "warrior_1", "quantity": null, "price": null, '
        '"priority": "medium", "short_reason": "ok"}'
    )
    with pytest.raises(InvalidAgentOutput, match="ACK"):
        parse_action(raw, condition=Condition.B)


def _back_from_a_trip(engine: Engine, account, resting: bool) -> None:
    """Put the adventurer in town, resting or rested."""
    account.expedition = None
    account.recovering_until_tick = engine.tick + (5 if resting else 0)


def test_a_resting_adventurer_trades_but_does_not_set_out(
    live_engine: Engine, config: SimulationConfig
) -> None:
    account = live_engine.state.account("warrior_1")
    fully_equip(account)
    for product in PROVISIONS:
        account.receive(product, 5)
    account.receive(IRON, 3)
    _back_from_a_trip(live_engine, account, resting=True)
    policy = policy_for(Role.ADVENTURER, config)
    rng = live_engine.random.agent("warrior_1")

    first = policy.decide(live_engine, account, rng)
    assert first.action is ActionType.SELL
    live_engine.submit_order("warrior_1", OrderSide.SELL, IRON, 3, "50")  # on the book now
    second = policy.decide(live_engine, account, rng)
    assert second.action is ActionType.WAIT  # nothing left to trade, still resting


def test_a_rested_adventurer_sets_out_with_orders_still_open(
    live_engine: Engine, config: SimulationConfig
) -> None:
    account = live_engine.state.account("warrior_1")
    fully_equip(account)
    for product in PROVISIONS:
        account.receive(product, 5)
    account.receive(IRON, 6)
    account.receive(Resource.LEATHER, 6)
    policy = policy_for(Role.ADVENTURER, config)
    rng = live_engine.random.agent("warrior_1")

    _back_from_a_trip(live_engine, account, resting=True)
    sells = orders_of(policy.decide(live_engine, account, rng))
    assert {o.target for o in sells if o.action is ActionType.SELL} == {"iron", "leather"}
    _back_from_a_trip(live_engine, account, resting=False)
    # Haul still unsold and orders open: it leaves anyway.
    action = policy.decide(live_engine, account, rng)
    assert action.action is ActionType.START_EXPEDITION


def test_an_adventurer_that_could_not_trade_while_resting_gets_one_trade_then_leaves(
    live_engine: Engine, config: SimulationConfig
) -> None:
    account = live_engine.state.account("warrior_1")
    fully_equip(account)
    for product in PROVISIONS:
        account.receive(product, 5)
    account.receive(IRON, 6)
    account.receive(Resource.LEATHER, 6)
    _back_from_a_trip(live_engine, account, resting=False)
    policy = policy_for(Role.ADVENTURER, config)
    rng = live_engine.random.agent("warrior_1")

    trade = orders_of(policy.decide(live_engine, account, rng))
    assert all(o.action is ActionType.SELL for o in trade)
    assert policy.decide(live_engine, account, rng).action is ActionType.START_EXPEDITION


def test_place_orders_executes_each_order_on_its_own(tmp_path: Path) -> None:
    """One bad order does not void the others."""
    from rpg_economy.agents import execute
    from rpg_economy.domain import RejectionCode

    engine, log = _executing_engine(tmp_path)
    engine.state.account("warrior_1").receive(IRON, 3)
    action = AgentAction(
        action=ActionType.PLACE_ORDERS,
        orders=(
            AgentAction(action=ActionType.SELL, target="iron", quantity=3, price=Decimal("12")),
            AgentAction(action=ActionType.SELL, target="wood", quantity=5, price=Decimal("8")),
            AgentAction(
                action=ActionType.BUY, target="health_potion", quantity=1, price=Decimal("15")
            ),
        ),
    )
    with log:
        outcome = execute(engine, "warrior_1", action)
    assert outcome.accepted and outcome.engine_operation and not outcome.wasted
    assert [r["accepted"] for r in outcome.order_results] == [True, False, True]
    assert outcome.order_results[1]["rejection"] == RejectionCode.INSUFFICIENT_INVENTORY.value
    sides = sorted(o.side.value for o in engine.market.open_orders_of("warrior_1"))
    assert sides == ["buy", "sell"]

    events = list(read_events(tmp_path / "run.jsonl"))
    assert sum(e.type is EventType.AGENT_OUTPUT for e in events) == 1
    assert [e.payload["order_index"] for e in events if e.type is EventType.ACTION_EXECUTED] == [
        0,
        2,
    ]
    assert [e.payload["order_index"] for e in events if e.type is EventType.ACTION_REJECTED] == [1]


def test_place_orders_with_nothing_accepted_is_a_wasted_activation(tmp_path: Path) -> None:
    from rpg_economy.agents import execute

    engine, log = _executing_engine(tmp_path)
    action = AgentAction(
        action=ActionType.PLACE_ORDERS,
        orders=(
            AgentAction(action=ActionType.SELL, target="wood", quantity=5, price=Decimal("8")),
            AgentAction(action=ActionType.SELL, target="herbs", quantity=5, price=Decimal("8")),
        ),
    )
    with log:
        outcome = execute(engine, "warrior_1", action)
    assert not outcome.accepted and outcome.wasted
    assert outcome.code is not None


def test_a_bundle_of_one_is_the_order_itself(config: SimulationConfig) -> None:
    from rpg_economy.agents.rule_based import RuleBasedPolicy

    single = AgentAction(action=ActionType.SELL, target="iron", quantity=1, price=Decimal("9"))
    assert RuleBasedPolicy._bundle([single], "x") is single
    assert RuleBasedPolicy._bundle([], "x") is None
    both = RuleBasedPolicy._bundle([single, single], "x")
    assert both.action is ActionType.PLACE_ORDERS and len(both.orders) == 2


def test_agents_never_place_more_orders_than_the_cap(
    live_engine: Engine, config: SimulationConfig
) -> None:
    cap = config.market.max_orders_per_activation
    _two_sided_books(live_engine)
    for agent_id in ("warrior_1", "smithing_1", "woodworking_1"):
        account = live_engine.state.account(agent_id)
        for resource in Resource:
            account.receive(resource, 3)
        action = policy_for(account.role, config).decide(
            live_engine, account, live_engine.random.agent(agent_id)
        )
        assert len(orders_of(action)) <= cap


def test_an_artisan_prices_a_unit_from_its_share_of_the_batch(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """One batch makes several potions: the cost floor is per potion, not per batch."""
    from rpg_economy.agents.rule_based import ArtisanPolicy

    policy = policy_for(Role.ARTISAN, config)
    assert isinstance(policy, ArtisanPolicy)
    recipe = config.economy.recipes[Product.MANA_POTION]
    assert recipe.output_quantity > 1
    batch = policy._input_cost(live_engine, Product.MANA_POTION)
    assert policy._unit_cost(live_engine, Product.MANA_POTION) == batch / recipe.output_quantity


def test_a_buyer_never_bids_past_the_ceiling_over_the_starting_price(
    live_engine: Engine, config: SimulationConfig
) -> None:
    """Escalation from a recent price that keeps rising would spiral without end."""
    from rpg_economy.agents.rule_based import MAX_BID_OVER_START, bid_price, starting_price

    good = Product.HEALTH_POTION
    ceiling = starting_price(live_engine, good) * (1 + MAX_BID_OVER_START)
    price = bid_price(live_engine, good, ceiling * 3, escalation=Decimal("0.6"))
    assert price <= ceiling.quantize(Decimal("0.01"))
