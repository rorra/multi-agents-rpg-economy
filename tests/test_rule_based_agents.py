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
