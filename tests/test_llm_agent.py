"""LLM-backed activations: prompts, parsing, retry policy, token buckets."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.agents.action import AgentAction
from rpg_economy.agents.llm_agent import MAX_ATTEMPTS, LLMAgentRunner
from rpg_economy.agents.observation import build_observation
from rpg_economy.agents.offline import json_action
from rpg_economy.agents.parsing import InvalidAgentOutput, parse_action
from rpg_economy.agents.prompts import (
    PROMPT_VERSION,
    SECTION_BULLETIN,
    SECTION_P2P,
    build_agent_prompt,
    prompt_hashes,
    render_world,
)
from rpg_economy.domain import (
    ActionType,
    AdventurerSubclass,
    Condition,
    MessageType,
    Priority,
    Product,
    Role,
    TokenBucket,
)
from rpg_economy.engine import Engine, build_world
from rpg_economy.llm import MockLLMClient, TokenLedger, always, scripted
from rpg_economy.logging_ import EventLogWriter, EventType, read_events
from rpg_economy.messaging import MessageBus
from rpg_economy.orchestration import build_directory
from rpg_economy.orchestration.global_view import build_global_summary
from tests.conftest import make_config, make_metadata


@pytest.fixture
def rig(tmp_path: Path):
    config = make_config(run={"condition": "B"})
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        messages = MessageBus(build_directory(engine))
        yield config, engine, messages


def runner(config, engine, messages, responder) -> tuple[LLMAgentRunner, TokenLedger]:
    ledger = TokenLedger()
    client = MockLLMClient(responder)
    return LLMAgentRunner(config, client, ledger, messages=messages), ledger


# ---------------------------------------------------------------------------
# Prompts and traits
# ---------------------------------------------------------------------------
def test_the_prompt_carries_all_four_traits(rig) -> None:
    """Design v1.0 s7 names exactly four trait dimensions."""
    config, engine, messages = rig
    observation = build_observation(config, engine, "woodworking_1", messages=messages)
    prompt = build_agent_prompt(observation, config)

    for trait in (
        "Risk tolerance",
        "Time orientation",
        "Commercial aggressiveness",
        "Adaptability",
    ):
        assert trait in prompt.system


def test_traits_differ_between_agents_of_different_roles(rig) -> None:
    config, engine, messages = rig
    artisan = build_observation(config, engine, "woodworking_1", messages=messages).traits
    warrior = build_observation(config, engine, "warrior_1", messages=messages).traits
    assert artisan != warrior


def test_traits_are_identical_across_reruns(rig) -> None:
    """Design v1.0 s12: same agents, same traits in matched runs."""
    config, engine, messages = rig
    first = build_observation(config, engine, "warrior_1", messages=messages).traits
    second = build_observation(make_config(), engine, "warrior_1", messages=messages).traits
    assert first == second


def test_the_prompt_shows_only_permitted_market_information(rig) -> None:
    """No other agent's gold, inventory or plans (design v1.0 s5.2)."""
    config, engine, messages = rig
    engine.state.account("warrior_1").gold = Decimal("4242.42")
    observation = build_observation(config, engine, "woodworking_1", messages=messages)
    prompt = build_agent_prompt(observation, config)

    assert "4242.42" not in prompt.user


def test_the_directory_shows_id_role_and_specialization_only(rig) -> None:
    config, engine, messages = rig
    observation = build_observation(config, engine, "woodworking_1", messages=messages)
    for entry in observation.directory:
        assert set(entry) == {"agent_id", "role", "specialization"}


def test_prompt_templates_are_versioned_and_hashed() -> None:
    """Design v1.0 s15: prompts recorded and versioned."""
    hashes = prompt_hashes()
    assert hashes["version"]
    assert "rules" in hashes and "action_schema" in hashes
    # Condition A's schema is hashed like the rest.
    assert "action_schema_a" in hashes
    assert PROMPT_VERSION == "2.61"  # IMPLEMENTATION.md section 4.8.1
    assert prompt_hashes() == prompt_hashes()


def test_the_world_rules_state_expiry_taxes_storage_and_the_floor() -> None:
    from rpg_economy.agents.prompts import render_world

    config = make_config()
    text = render_world(config)
    assert f"orders expire after {config.market.order_expiry_ticks} ticks" in text
    taxes = config.economy.taxes
    assert (
        f"profit tax, paid by the seller of a finished product: {taxes.profit_rate * 100:g}% "
        "of what the sale brings in above the cost of its inputs" in text
    )
    assert f"wealth tax: every {taxes.wealth_every_ticks} ticks" in text
    assert f"of gold above {taxes.wealth_exemption:g}" in text
    assert (
        "storage, charged at the start of every tick per unit of goods held: "
        "adventurer 0.03 gold, artisan 0.03 gold" in text
    )
    assert (
        "storage and the profit tax never take your gold below your starting gold "
        "(adventurer 100, artisan 150)" in text
    )
    assert "never charged below" not in text


def test_the_costs_state_their_period_in_ticks_from_the_config() -> None:
    """Storage and owed tax are charged as a cycle opens; agents count ticks."""
    from rpg_economy.agents.prompts import render_world

    config = make_config()
    assert config.time.ticks_per_cycle == 1
    assert "gold at the start of every tick" in render_world(config)

    longer = config.model_copy(
        update={"time": config.time.model_copy(update={"ticks_per_cycle": 5})}
    )
    text = render_world(longer)
    assert "collected from free gold at the start of every 5 ticks" in text
    assert "storage, charged at the start of every 5 ticks per unit of goods held" in text


def test_b_and_c_are_told_what_messages_do_and_that_answers_are_final() -> None:
    from rpg_economy.agents.prompts import ACTION_SCHEMA

    assert "Messages move no gold and no goods" in ACTION_SCHEMA
    assert "ACK and REJECT are never answered" in ACTION_SCHEMA
    assert "places your matching order" in ACTION_SCHEMA
    assert "An ACK to your REQUEST or OFFER means the other agent placed its matching" in (
        ACTION_SCHEMA
    )
    assert "expires after the same number of ticks as an order" in ACTION_SCHEMA


def test_an_adventurer_is_told_which_zones_exist(rig) -> None:
    config, engine, messages = rig
    observation = build_observation(config, engine, "warrior_1", messages=messages)
    prompt = build_agent_prompt(observation, config)
    assert "green_hollow" in prompt.system


# ---------------------------------------------------------------------------
# Retry-then-WAIT (design v1.0 s7)
# ---------------------------------------------------------------------------
def test_an_adventurer_is_told_its_chance_per_zone(rig) -> None:
    """The chance the engine would use if it set out now, per zone."""
    config, engine, messages = rig
    warrior = build_observation(config, engine, "warrior_1", messages=messages)
    prompt = build_agent_prompt(warrior, config)
    odds = engine.expedition_odds("warrior_1")
    for zone, chance in odds.items():
        assert f'"{zone}":"{chance.quantize(Decimal("0.01"))}"' in prompt.sections["state"]
    smith = build_observation(config, engine, "smithing_1", messages=messages)
    assert '"your_success_chance":null' in build_agent_prompt(smith, config).sections["state"]


def test_an_adventurer_is_told_what_stops_it_setting_out(rig) -> None:
    config, engine, messages = rig
    account = engine.state.account("warrior_1")

    def status() -> str:
        observation = build_observation(config, engine, "warrior_1", messages=messages)
        return build_agent_prompt(observation, config).sections["state"]

    assert "Setting out: ready now" in status()
    account.equipment.clear()
    # Gear is optional, but what it lacks and would pay is named (2.34).
    assert (
        "Setting out: ready now, but you do not carry gear that pays for itself: nobody offers a"
        in status()
    )
    assert "bid first and set out next" in status()
    # The price to bid is the item's own, not its cost a trip (2.35).
    weapon_guide = config.economy.initial_reference_price["melee_weapon"]
    assert f"lasts 4 trips): bid about {weapon_guide:.2f} for one" in status()
    from rpg_economy.domain import OrderSide

    engine.state.account("smithing_1").receive(Product.MELEE_WEAPON, 1)
    engine.submit_order("smithing_1", OrderSide.SELL, Product.MELEE_WEAPON, 1, "39.00")
    assert "melee_weapon offered at 39.00 adds about" in status()
    account.recovering_until_tick = engine.tick + 3
    resting = status()
    assert f"resting until tick {engine.tick + 3} (it is tick {engine.tick})" in resting
    assert "trade meanwhile; meanwhile, " in resting and "melee_weapon offered at" in resting
    smith = build_observation(config, engine, "smithing_1", messages=messages)
    assert "Setting out" not in build_agent_prompt(smith, config).sections["state"]


def test_signals_name_a_short_good_and_an_unfilled_bid(rig) -> None:
    """Pilot: shown only the raw book, agents re-bid the old price forever."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    alchemist = engine.state.account("alchemy_1")
    alchemist.gold += Decimal("100")
    engine.submit_order("alchemy_1", OrderSide.BUY, Resource.HERBS, 2, "6.00")
    for _ in range(config.market.order_expiry_ticks + 1):
        engine.run_cycle()
    engine.submit_order("alchemy_2", OrderSide.BUY, Resource.HERBS, 1, "6.00")

    observation = build_observation(config, engine, "alchemy_1", messages=messages)
    herbs = next(line for line in observation.signals if line.startswith("- herbs:"))
    assert "SHORT: buyers bid 6.00 and nothing is offered" in herbs
    assert "your last 1 bid(s) at 6.00 expired with nothing bought" in herbs
    market = build_agent_prompt(observation, config).sections["market"]
    assert market.startswith("Market signals (from the book and your own orders):")
    # An adventurer cannot bid for a raw material: it is told only the seller's half.
    ranger = build_observation(config, engine, "ranger_1", messages=messages).signals
    assert (
        "- herbs (you only sell it): SHORT: buyers bid 6.00 and nothing is offered, so a "
        "seller can ask about 6.60"
    ) in ranger


def test_an_artisan_is_told_it_holds_what_a_buyer_waits_for(rig) -> None:
    """Pilot: a woodworker kept a staff 100 ticks while both mages bid for one."""
    from rpg_economy.domain import OrderSide

    config, engine, messages = rig
    engine.state.account("mage_1").gold += Decimal("100")
    engine.submit_order("mage_1", OrderSide.BUY, Product.STAFF_FOCUS, 1, "60.00")
    engine.state.account("woodworking_1").receive(Product.STAFF_FOCUS, 1)

    signals = build_observation(config, engine, "woodworking_1", messages=messages).signals
    staff = next(line for line in signals if line.startswith("- staff_focus:"))
    assert "you hold 1; buyers are waiting and nothing else is offered" in staff
    craft = next(line for line in signals if line.startswith("- craft staff_focus:"))
    assert "; you already hold 1 unsold; one costs about" in craft
    # A buyer of the good is not told it holds it as stock to sell.
    mage = build_observation(config, engine, "mage_2", messages=messages).signals
    assert not any("you hold" in line for line in mage)


def test_an_artisan_is_told_its_input_cost_and_a_lowball_bid_is_no_shortage(rig) -> None:
    """Pilot: woodworkers sold bows worth 38 in inputs to a 15.50 bid."""
    from rpg_economy.agents.observation import unit_cost
    from rpg_economy.domain import OrderSide

    config, engine, messages = rig
    engine.state.account("ranger_1").gold += Decimal("100")
    engine.submit_order("ranger_1", OrderSide.BUY, Product.BOW_GEAR, 1, "15.50")
    engine.state.account("mage_1").gold += Decimal("100")
    engine.submit_order("mage_1", OrderSide.BUY, Product.STAFF_FOCUS, 1, "90.00")
    account = engine.state.account("woodworking_1")
    account.receive(Product.BOW_GEAR, 1)
    account.receive(Product.STAFF_FOCUS, 1)

    # No trades yet: the inputs at their starting reference prices.
    prices = config.economy.initial_reference_price
    bow = config.economy.recipes[Product.BOW_GEAR]
    expected = sum(Decimal(str(prices[r.value])) * q for r, q in bow.inputs.items())
    assert unit_cost(config, engine.market, Product.BOW_GEAR) == expected / bow.output_quantity

    signals = build_observation(config, engine, "woodworking_1", messages=messages).signals
    # The staff a buyer pays well for comes first; the lowball bow bid is no shortage.
    assert signals[0].startswith("- staff_focus: one costs about")
    assert "you hold 1; buyers are waiting" in signals[0]
    bow_line = next(line for line in signals if line.startswith("- bow_gear:"))
    assert "the only bid is 15.50, below what one costs you: selling there loses gold" in bow_line
    assert "SHORT" not in bow_line and "you hold" not in bow_line
    craft = next(line for line in signals if line.startswith("- craft bow_gear:"))
    assert "below what one costs you" in craft and "sells at once" not in craft


def test_an_artisans_headline_names_a_sale_ready_and_unsold_stock(rig) -> None:
    """Replayed at real moments: in the signals alone a held staff sold 1 time in 42."""
    from rpg_economy.domain import OrderSide

    config, engine, messages = rig
    engine.state.account("mage_1").gold += Decimal("100")
    engine.submit_order("mage_1", OrderSide.BUY, Product.STAFF_FOCUS, 1, "90.00")
    account = engine.state.account("woodworking_1")
    account.receive(Product.STAFF_FOCUS, 1)
    account.receive(Product.BOW_GEAR, 4)

    observation = build_observation(config, engine, "woodworking_1", messages=messages)
    assert observation.status[0].startswith("Ready to sell: staff_focus x1: buyers bid 90.00")
    assert observation.status[-1].startswith("Unsold: bow_gear: you already hold 4")
    state = build_agent_prompt(observation, config).sections["state"]
    assert state.startswith("Your state:\nReady to sell: staff_focus") and "Unsold: bow" in state

    # A bid below the input cost is no sale ready; an adventurer gets no headline.
    engine.submit_order("mage_1", OrderSide.BUY, Product.BOW_GEAR, 1, "5.00")
    observation = build_observation(config, engine, "woodworking_1", messages=messages)
    assert not any("bow_gear" in line for line in observation.status)
    warrior = build_observation(config, engine, "warrior_1", messages=messages).status
    assert all(line.startswith(("Gear:", "Zones now:")) for line in warrior)  # no artisan lines


def test_units_on_the_artisans_own_sell_orders_count_as_unsold(rig) -> None:
    """A smith with 6 weapons on its own unfilled orders was never told
    and crafted more, while buyers bid for armor."""
    from rpg_economy.domain import OrderSide

    config, engine, messages = rig
    smith = engine.state.account("smithing_1")
    smith.receive(Product.MELEE_WEAPON, 4)
    engine.submit_order("smithing_1", OrderSide.SELL, Product.MELEE_WEAPON, 4, "60.00")
    engine.state.account("ranger_1").gold += Decimal("100")
    engine.submit_order("ranger_1", OrderSide.BUY, Product.ARMOR, 1, "66.00")

    status = build_observation(config, engine, "smithing_1", messages=messages).status
    assert (
        "Unsold: melee_weapon: you already hold 4 and no one is bidding; you can craft "
        "more once fewer than 2 remain; craft armor (buyers bid 66.00) instead" in status
    )


def test_an_artisans_headline_names_wanted_products_and_unfilled_inputs(rig) -> None:
    """In the signals alone, smiths crafted what nobody bid for and alchemists
    re-bid the old price; as headlines, both were acted on."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    smith = engine.state.account("smithing_1")
    smith.receive(Resource.IRON, 4)
    smith.receive(Resource.LEATHER, 4)
    engine.state.account("ranger_1").gold += Decimal("100")
    engine.submit_order("ranger_1", OrderSide.BUY, Product.ARMOR, 1, "52.00")
    status = build_observation(config, engine, "smithing_1", messages=messages).status
    assert (
        "Wanted: armor: buyers bid 52.00 and nobody sells it; your inputs allow 3 run(s) now"
        in status
    )

    engine.state.account("alchemy_1").gold += Decimal("100")
    engine.submit_order("alchemy_1", OrderSide.BUY, Resource.HERBS, 2, "6.00")
    for _ in range(config.market.order_expiry_ticks + 1):
        engine.run_cycle()
    status = build_observation(config, engine, "alchemy_1", messages=messages).status
    assert any(
        line.startswith("Buying: herbs: your last 1 bid(s) at 6.00 expired with nothing bought")
        and line.endswith("bid about 6.60 to get any")
        for line in status
    )


def test_an_artisan_is_told_when_an_input_costs_more_than_its_products_pay(rig) -> None:
    """Smiths bid iron from 10.5 to 30, paying 58 for a weapon sold at 48-53."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    # Its own bid lapsed: short of iron, it is not told to step down.
    engine.submit_order("smithing_1", OrderSide.BUY, Resource.IRON, 1, "10.00")
    for _ in range(config.market.order_expiry_ticks + 1):
        engine.run_cycle()
    engine.state.account("warrior_1").receive(Resource.IRON, 1)
    engine.state.account("smithing_2").gold += Decimal("100")
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.IRON, 1, "23.89")
    engine.submit_order("smithing_2", OrderSide.BUY, Resource.IRON, 1, "23.89")
    status = build_observation(config, engine, "smithing_1", messages=messages).status
    too_dear = [line for line in status if line.startswith("Too dear: iron:")]
    assert len(too_dear) == 1
    assert (
        "an armor or melee_weapon made from it sells for less than 15% over what its "
        "inputs cost" in too_dear[0]
    )
    # Leather is valued against iron's typical price, not its bid-up last trade.
    assert not any(line.startswith("Too dear: leather") for line in status)


def test_the_input_ceiling_follows_the_products_price(rig) -> None:
    """A dearer product lifts what its input is worth, so a shock can raise both."""
    from rpg_economy.agents.observation import input_ceiling
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    alchemist = engine.state.account("alchemy_1")
    before, _ = input_ceiling(config, engine, alchemist, Resource.HERBS)
    engine.state.account("ranger_1").gold += Decimal("100")
    engine.submit_order("ranger_1", OrderSide.BUY, Product.MANA_POTION, 1, "40.00")
    after, products = input_ceiling(config, engine, alchemist, Resource.HERBS)
    assert after > before
    assert Product.HEALTH_POTION in products and Product.MANA_POTION in products


def test_an_input_with_nobody_offering_is_bid_at_its_last_price(rig) -> None:
    """In the herb shock alchemists were told to bid 6.00 for herbs last at 9.80."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    engine.state.account("warrior_1").receive(Resource.HERBS, 1)
    engine.state.account("alchemy_2").gold += Decimal("100")
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.HERBS, 1, "9.80")
    engine.submit_order("alchemy_2", OrderSide.BUY, Resource.HERBS, 1, "9.80")
    alchemist = engine.state.account("alchemy_1")
    alchemist.inventory[Resource.HERBS] = 0
    engine.state.account("mage_1").gold += Decimal("100")
    engine.submit_order("mage_1", OrderSide.BUY, Product.HEALTH_POTION, 1, "12.10")
    # Still selling, it is bid halfway back to its 6.00 typical price.
    status = build_observation(config, engine, "alchemy_1", messages=messages).status
    assert any("1 herbs (nobody offers any: bid about 7.90)" in line for line in status)
    # Once its own bid there lapses, at its last price.
    alchemist.gold += Decimal("100")
    engine.submit_order("alchemy_1", OrderSide.BUY, Resource.HERBS, 1, "7.90")
    for _ in range(config.market.recent_sale_ticks + 1):
        engine.run_cycle()
    engine.submit_order("mage_1", OrderSide.BUY, Product.HEALTH_POTION, 1, "12.10")
    status = build_observation(config, engine, "alchemy_1", messages=messages).status
    assert any("1 herbs (nobody offers any: bid about 9.80)" in line for line in status)


def test_a_raw_material_still_selling_above_its_typical_price_is_bid_down(rig) -> None:
    """Told to bid the last sale while herbs kept selling, alchemists held
    herbs at twice their typical price for 100 ticks with 17 left unsold."""
    from rpg_economy.agents.observation import market_signals
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    engine.state.account("warrior_1").receive(Resource.HERBS, 1)
    engine.state.account("alchemy_2").gold += Decimal("100")
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.HERBS, 1, "11.50")
    engine.submit_order("alchemy_2", OrderSide.BUY, Resource.HERBS, 1, "11.50")
    engine.submit_order("alchemy_2", OrderSide.BUY, Resource.HERBS, 1, "11.67")
    alchemist = engine.state.account("alchemy_1")
    (line,) = [
        x for x in market_signals(config, engine, alchemist, [Resource.HERBS]) if "herbs" in x
    ]
    assert "a buyer who wants it can bid about 8.84, a step toward its 6.00 typical" in line
    assert "a seller can ask about 11.67" in line

    # A buyer whose own bids lapse is short of it: no step down.
    alchemist.gold += Decimal("100")
    engine.submit_order("alchemy_1", OrderSide.BUY, Resource.HERBS, 1, "11.00")
    for _ in range(config.market.order_expiry_ticks + 1):
        engine.run_cycle()
    engine.state.account("warrior_1").receive(Resource.HERBS, 1)
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.HERBS, 1, "11.50")
    engine.submit_order("alchemy_2", OrderSide.BUY, Resource.HERBS, 2, "11.50")
    (line,) = [
        x for x in market_signals(config, engine, alchemist, [Resource.HERBS]) if "herbs" in x
    ]
    assert "a buyer who wants it can bid about 11.50, and" in line


def test_a_product_is_offered_at_its_guide_margin_over_its_inputs_now(rig) -> None:
    """Offered at their typical price or cost, potions sold at cost
    through the herb shock."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    engine.state.account("warrior_1").receive(Resource.HERBS, 1)
    engine.state.account("alchemy_2").gold += Decimal("100")
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.HERBS, 1, "12.00")
    engine.submit_order("alchemy_2", OrderSide.BUY, Resource.HERBS, 1, "12.00")
    engine.state.account("alchemy_1").receive(Product.HEALTH_POTION, 1)
    status = build_observation(config, engine, "alchemy_1", messages=messages).status
    (line,) = [x for x in status if x.startswith("Not offered: health_potion")]
    assert any(x.startswith("Too dear: herbs: it last sold at 12.00") for x in status)
    # Guide 10.00 over 6.00 of herbs: herbs at 12.00 make it 20.00.
    assert "offer it at about 20.00, its guide margin over the 12.00 its inputs cost now" in line


def test_an_input_bid_suggestion_never_falls_below_the_artisans_own_failed_bids(rig) -> None:
    """Told "bid about 6.82" after its 7.50 expired, an alchemist went back to it."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    engine.state.account("warrior_1").receive(Resource.HERBS, 1)
    engine.state.account("alchemy_2").gold += Decimal("100")
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.HERBS, 1, "6.82")
    engine.submit_order("alchemy_2", OrderSide.BUY, Resource.HERBS, 1, "6.82")
    alchemist = engine.state.account("alchemy_1")
    alchemist.inventory[Resource.HERBS] = 0
    # Herbs stopped selling: no sale within recent_sale_ticks.
    for _ in range(config.market.recent_sale_ticks - config.market.order_expiry_ticks):
        engine.run_cycle()
    engine.submit_order("alchemy_1", OrderSide.BUY, Resource.HERBS, 1, "7.50")
    for _ in range(config.market.order_expiry_ticks + 1):
        engine.run_cycle()
    engine.state.account("mage_1").gold += Decimal("100")
    engine.submit_order("mage_1", OrderSide.BUY, Product.HEALTH_POTION, 1, "9.40")
    status = build_observation(config, engine, "alchemy_1", messages=messages).status
    assert any("1 herbs (nobody offers any: bid about 8.25)" in line for line in status)
    assert any(line.startswith("Buying: herbs:") and "bid about 8.25" in line for line in status)

    engine.state.account("mage_2").receive(Resource.HERBS, 3)
    engine.submit_order("mage_2", OrderSide.SELL, Resource.HERBS, 3, "7.42")
    status = build_observation(config, engine, "alchemy_1", messages=messages).status
    assert any(
        line.startswith("Buying: herbs: your last 1 bid(s) at 7.50 expired with nothing bought")
        and line.endswith("it is offered at 7.42: a bid at 7.42 buys now")
        for line in status
    )


def test_a_lapsed_bid_for_a_good_still_selling_is_not_stepped_up(rig) -> None:
    """Smiths' bids lapsed between adventurers' trips and each lapse
    stepped them 10% up: iron rose from 11.5 to 16.8 with more gathered than used."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    engine.state.account("warrior_1").receive(Resource.IRON, 1)
    engine.state.account("smithing_2").gold += Decimal("100")
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.IRON, 1, "11.68")
    engine.submit_order("smithing_2", OrderSide.BUY, Resource.IRON, 1, "11.68")
    engine.state.account("smithing_1").gold += Decimal("100")
    engine.submit_order("smithing_1", OrderSide.BUY, Resource.IRON, 2, "11.68")
    for _ in range(config.market.order_expiry_ticks + 1):
        engine.run_cycle()
    status = build_observation(config, engine, "smithing_1", messages=messages).status
    buying = [line for line in status if line.startswith("Buying: iron:")]
    assert len(buying) == 1
    assert "sellers come back between trips: bid about 11.68" in buying[0]


def test_a_bid_with_nothing_offered_is_not_a_shortage_when_the_good_sold_lately(rig) -> None:
    """Two buyers for leather, each told to bid a step above the other,
    took it from 9.1 to 12.8 with as much gathered as used."""
    from rpg_economy.agents.observation import market_signals
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    wood = engine.state.account("woodworking_1")
    engine.state.account("warrior_1").receive(Resource.LEATHER, 1)
    engine.state.account("smithing_2").gold += Decimal("100")
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.LEATHER, 1, "11.00")
    engine.submit_order("smithing_2", OrderSide.BUY, Resource.LEATHER, 1, "11.00")
    engine.state.account("smithing_1").gold += Decimal("100")
    engine.submit_order("smithing_1", OrderSide.BUY, Resource.LEATHER, 1, "10.98")
    view = " ".join(market_signals(config, engine, wood, [Resource.LEATHER]))
    assert "SHORT: buyers bid 10.98" not in view
    # Above its 9.00 guide it is bid a step down, never up.
    assert "a buyer who wants it can bid about 9.90, a step toward its 9.00 typical" in view
    assert "a seller can ask about 11.00" in view

    for _ in range(config.market.recent_sale_ticks + 1):
        engine.run_cycle()
    engine.submit_order("smithing_1", OrderSide.BUY, Resource.LEATHER, 1, "10.98")
    view = " ".join(market_signals(config, engine, wood, [Resource.LEATHER]))
    assert "SHORT: buyers bid 10.98 and nothing is offered" in view


def test_an_artisan_sees_first_the_product_a_buyer_waits_for(rig) -> None:
    """Pilot: smiths crafted armor, listed first, while warriors waited for weapons."""
    from rpg_economy.domain import OrderSide

    config, engine, messages = rig
    engine.state.account("warrior_1").gold += Decimal("100")
    engine.submit_order("warrior_1", OrderSide.BUY, Product.MELEE_WEAPON, 1, "40.00")
    engine.state.account("smithing_2").receive(Product.ARMOR, 1)
    engine.submit_order("smithing_2", OrderSide.SELL, Product.ARMOR, 1, "51.00")

    signals = build_observation(config, engine, "smithing_1", messages=messages).signals
    crafts = [line for line in signals if line.startswith("- craft ")]
    assert crafts[0].startswith("- craft melee_weapon:")
    assert "one crafted now sells at once" in crafts[0]
    assert crafts[-1].startswith("- craft armor:")
    assert "another would sit unsold" in crafts[-1]
    adventurer = build_observation(config, engine, "warrior_2", messages=messages).signals
    assert not any(line.startswith("- craft ") for line in adventurer)


def test_the_world_rules_state_the_gear_gold_bonus(rig) -> None:
    config, engine, messages = rig
    warrior = build_observation(config, engine, "warrior_1", messages=messages)
    prompt = build_agent_prompt(warrior, config)
    assert "required_to_set_out" not in prompt.sections["state"]
    assert "Gear gold: a successful trip's gold is raised" in prompt.system
    assert "melee_weapon +80%" in render_world(config)


def test_gear_lines_state_what_each_missing_item_adds_and_costs(rig) -> None:
    """2.34: optional gear, priced against what it adds to a trip, as facts."""
    from rpg_economy.domain import OrderSide

    config, engine, messages = rig
    account = engine.state.account("warrior_1")
    account.equipment.clear()
    engine.state.account("smithing_1").receive(Product.MELEE_WEAPON, 1)
    engine.submit_order("smithing_1", OrderSide.SELL, Product.MELEE_WEAPON, 1, "40.00")
    status = build_observation(config, engine, "warrior_1", messages=messages).status
    weapon = next(line for line in status if line.startswith("Gear: melee_weapon"))
    assert "(offered at 40.00; lasts 4 trips, 10.00 a trip) adds about" in weapon
    assert "in green_hollow" in weapon and "in arcane_deeps" in weapon
    armor = next(line for line in status if line.startswith("Gear: armor"))
    armor_guide = config.economy.initial_reference_price["armor"]
    assert f"nobody offers one; guide price {armor_guide:.2f}" in armor
    # Held, setting out equips it (2.36); worn, too, it is not listed.
    account.receive(Product.MELEE_WEAPON, 1)
    status = build_observation(config, engine, "warrior_1", messages=messages).status
    assert not any(line.startswith("Gear: melee_weapon") for line in status)
    account.equipment[Product.MELEE_WEAPON] = 4
    account.inventory.pop(Product.MELEE_WEAPON)
    status = build_observation(config, engine, "warrior_1", messages=messages).status
    assert not any("melee_weapon" in line for line in status)
    smith = build_observation(config, engine, "smithing_1", messages=messages).status
    assert not any(line.startswith("Gear:") for line in smith)


def test_an_adventurer_is_told_its_expedition_gear(rig) -> None:
    """Each item of the subclass's gear adds its own bonus (design s5.3), so say which."""
    config, engine, messages = rig
    mage = build_observation(config, engine, "mage_1", messages=messages)
    prompt = build_agent_prompt(mage, config)
    mana = config.expeditions.gear[AdventurerSubclass.MAGE][Product.MANA_POTION]
    assert f'"mana_potion":"{mana}"' in prompt.sections["state"]
    # Named expedition_gear, the table read as gear held.
    assert "gear_if_carried lists every item your subclass can use" in prompt.system
    assert "it is not what you hold" in prompt.system

    artisan = build_observation(config, engine, "woodworking_1", messages=messages)
    assert artisan.expedition_gear == {}
    assert '"gear_if_carried":null' in build_agent_prompt(artisan, config).sections["state"]


def test_a_broken_main_item_heads_the_adventurers_state(rig) -> None:
    """Warriors whose weapon broke set out without one."""
    config, engine, messages = rig
    warrior = engine.state.account("warrior_1")
    warrior.equipment.pop(Product.MELEE_WEAPON, None)
    status = build_observation(config, engine, "warrior_1", messages=messages).status
    assert status[0].startswith("Broken: your melee_weapon broke and you carry none")
    assert "before you set out" in status[0]

    # Gone once it bids for one.
    from rpg_economy.domain import OrderSide

    warrior.gold += Decimal("60")
    engine.submit_order("warrior_1", OrderSide.BUY, Product.MELEE_WEAPON, 1, "48.00")
    status = build_observation(config, engine, "warrior_1", messages=messages).status
    assert not any(x.startswith("Broken") for x in status)


def test_an_artisan_is_told_to_offer_a_product_nobody_offers_or_bids_for(rig) -> None:
    """Smiths held weapons nobody bid for and never offered them."""
    config, engine, messages = rig
    engine.state.account("smithing_1").receive(Product.MELEE_WEAPON, 1)
    status = build_observation(config, engine, "smithing_1", messages=messages).status
    (line,) = [x for x in status if x.startswith("Not offered")]
    guide = config.economy.initial_reference_price["melee_weapon"]
    assert f"melee_weapon x1: nobody offers or bids for one; offer it at about {guide:.2f}" in line


def test_a_valid_first_response_is_used(rig) -> None:
    config, engine, messages = rig
    agent, _ = runner(config, engine, messages, always(json_action(action="wait")))
    result = agent.activate(engine, "woodworking_1")

    assert result.attempts == 1
    assert result.invalid_outputs == 0
    assert result.action.action is ActionType.WAIT


def test_one_retry_is_allowed_after_invalid_output(rig) -> None:
    config, engine, messages = rig
    engine.state.account("woodworking_1").receive_gold = None
    agent, _ = runner(
        config,
        engine,
        messages,
        scripted(["not json at all", json_action(action="wait", short_reason="recovered")]),
    )
    result = agent.activate(engine, "woodworking_1")

    assert result.attempts == 2
    assert result.invalid_outputs == 1
    assert result.action.short_reason == "recovered"


def test_two_invalid_outputs_waste_the_activation_and_wait(rig) -> None:
    config, engine, messages = rig
    agent, _ = runner(config, engine, messages, always("still not json"))
    result = agent.activate(engine, "woodworking_1")

    assert result.attempts == MAX_ATTEMPTS
    assert result.invalid_outputs == MAX_ATTEMPTS
    assert result.action.action is ActionType.WAIT
    assert result.wasted


def test_there_is_never_a_second_retry(rig) -> None:
    config, engine, messages = rig
    client = MockLLMClient(always("nope"))
    agent = LLMAgentRunner(config, client, TokenLedger(), messages=messages)
    agent.activate(engine, "woodworking_1")
    assert client.calls == MAX_ATTEMPTS


def test_invalid_output_is_logged_with_its_reason(rig) -> None:
    config, engine, messages = rig
    agent, _ = runner(config, engine, messages, always("garbage"))
    agent.activate(engine, "woodworking_1")
    engine.log.close()

    events = list(read_events(engine.log.path))
    invalid = [e for e in events if e.type is EventType.AGENT_OUTPUT_INVALID]
    assert len(invalid) == MAX_ATTEMPTS
    assert invalid[0].payload["reason"]
    assert any(e.type is EventType.ACTIVATION_WASTED for e in events)


# ---------------------------------------------------------------------------
# The wasted-activation predicate (design v1.0 s11.2)
# ---------------------------------------------------------------------------
def test_wait_is_a_wasted_activation(rig) -> None:
    config, engine, messages = rig
    agent, _ = runner(config, engine, messages, always(json_action(action="wait")))
    assert agent.activate(engine, "woodworking_1").wasted


def test_a_placed_order_is_not_wasted(rig) -> None:
    config, engine, messages = rig
    engine.state.account("woodworking_1").receive(
        __import__("rpg_economy.domain", fromlist=["Resource"]).Resource.IRON, 5
    )
    agent, _ = runner(
        config,
        engine,
        messages,
        always(json_action(action="sell", target="iron", quantity=2, price=10.0)),
    )
    assert not agent.activate(engine, "woodworking_1").wasted


def test_a_message_only_activation_is_not_wasted(rig) -> None:
    """An activation that only communicated still did something."""
    config, engine, messages = rig
    agent, _ = runner(
        config,
        engine,
        messages,
        always(
            json_action(
                action="send_message",
                message_type="REQUEST",
                target="iron",
                quantity=1,
                price="5",
                recipient="warrior_1",
            )
        ),
    )
    result = agent.activate(engine, "woodworking_1")
    assert result.outcome.message_emitted
    assert not result.wasted


def test_a_rejected_action_is_wasted(rig) -> None:
    config, engine, messages = rig
    agent, _ = runner(
        config,
        engine,
        messages,
        always(json_action(action="sell", target="iron", quantity=99, price=10.0)),
    )
    result = agent.activate(engine, "woodworking_1")
    assert not result.outcome.accepted
    assert result.wasted


# ---------------------------------------------------------------------------
# Token buckets
# ---------------------------------------------------------------------------
def test_agent_tokens_are_charged_to_the_agent_bucket(rig) -> None:
    config, engine, messages = rig
    agent, ledger = runner(config, engine, messages, always(json_action(action="wait")))
    agent.activate(engine, "woodworking_1")

    assert ledger.total(TokenBucket.AGENT) > 0
    assert ledger.total(TokenBucket.DIRECTOR) == 0
    assert ledger.total(TokenBucket.BULLETIN) == 0


def test_p2p_content_is_charged_to_its_own_bucket(rig) -> None:
    """Design v1.0 s8.4: P2P content inside a prompt is billed separately."""
    config, engine, messages = rig
    messages.send(
        sender_id="warrior_1",
        recipient_id="woodworking_1",
        message_type=MessageType.REQUEST,
        tick=0,
        payload={"good": "iron", "quantity": 5},
    )
    agent, ledger = runner(config, engine, messages, always(json_action(action="wait")))
    agent.activate(engine, "woodworking_1")

    assert ledger.input_tokens(TokenBucket.P2P_CONTENT) > 0


def test_no_p2p_tokens_when_the_inbox_is_empty(rig) -> None:
    config, engine, messages = rig
    agent, ledger = runner(config, engine, messages, always(json_action(action="wait")))
    agent.activate(engine, "woodworking_1")
    assert ledger.input_tokens(TokenBucket.P2P_CONTENT) == 0


@pytest.mark.invariant
def test_bulletin_tokens_are_charged_only_on_inclusion(tmp_path: Path) -> None:
    """Design v1.0 s3.3: counted only when actually included in a prompt."""
    ledgers = {}
    for condition in (Condition.B, Condition.C):
        config = make_config(run={"condition": condition.value})
        state = build_world(config)
        with EventLogWriter(tmp_path / f"{condition.value}.jsonl", make_metadata(config)) as log:
            engine = Engine(config, state, log)
            # The canonical generator, not a hand-made dict: the renderer is the
            # one A and C share, so it only accepts a real global summary.
            engine.publish_bulletin(build_global_summary(config, engine))
            messages = MessageBus(build_directory(engine))
            agent, ledger = runner(config, engine, messages, always(json_action(action="wait")))
            agent.activate(engine, "woodworking_1")
            ledgers[condition] = ledger

    # Published in both, but only Condition C puts it in a prompt.
    assert ledgers[Condition.B].input_tokens(TokenBucket.BULLETIN) == 0
    assert ledgers[Condition.C].input_tokens(TokenBucket.BULLETIN) > 0


def test_buckets_reconcile_with_the_reported_usage(rig) -> None:
    config, engine, messages = rig
    messages.send(
        sender_id="warrior_1",
        recipient_id="woodworking_1",
        message_type=MessageType.INFORM,
        tick=0,
        payload={"note": "iron is scarce"},
    )
    client = MockLLMClient(always(json_action(action="wait")))
    ledger = TokenLedger()
    agent = LLMAgentRunner(config, client, ledger, messages=messages)
    agent.activate(engine, "woodworking_1")
    engine.log.close()

    calls = [e for e in read_events(engine.log.path) if e.type is EventType.LLM_CALL]
    assert len(calls) == 1
    reported_input = calls[0].payload["input_tokens"]
    assert sum(calls[0].payload["input_by_bucket"].values()) == reported_input
    assert ledger.input_tokens() == reported_input


def test_every_llm_call_is_logged(rig) -> None:
    config, engine, messages = rig
    agent, _ = runner(config, engine, messages, always(json_action(action="wait")))
    agent.activate(engine, "woodworking_1")
    engine.log.close()

    calls = [e for e in read_events(engine.log.path) if e.type is EventType.LLM_CALL]
    assert len(calls) == 1
    assert calls[0].payload["model"].startswith("mock")


# ---------------------------------------------------------------------------
# Parsing round-trips
# ---------------------------------------------------------------------------
def test_action_round_trips_through_json() -> None:
    original = AgentAction(
        action=ActionType.BUY,
        target="iron",
        quantity=4,
        price=Decimal("11.25"),
        priority=Priority.HIGH,
        short_reason="stock up",
    )
    parsed = parse_action(json_action(**original.to_payload()), condition=Condition.B)
    assert parsed.action is original.action
    assert parsed.target == original.target
    assert parsed.quantity == original.quantity
    assert parsed.price == original.price
    assert parsed.priority is original.priority


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json",
        '{"action": "teleport"}',
        '{"action": "buy", "target": "iron"}',
        '{"action": "buy", "target": "iron", "quantity": -1, "price": 5}',
        '{"action": "produce"}',
        '{"action": "start_expedition"}',
    ],
)
def test_malformed_output_is_rejected(text: str) -> None:
    with pytest.raises(InvalidAgentOutput):
        parse_action(text, condition=Condition.B)


def test_fenced_and_embedded_json_are_both_accepted() -> None:
    fenced = '```json\n{"action":"wait","short_reason":"ok"}\n```'
    embedded = 'I will wait. {"action":"wait","short_reason":"ok"} Done.'
    for text in (fenced, embedded):
        assert parse_action(text, condition=Condition.B).action is ActionType.WAIT


@pytest.mark.invariant
def test_condition_a_rejects_peer_messages(rig) -> None:
    """Design s3.1: no P2P in Condition A, via the ordinary retry path."""
    with pytest.raises(InvalidAgentOutput, match="Condition A"):
        parse_action(
            json_action(
                action="send_message",
                message_type="REQUEST",
                target="iron",
                quantity=1,
                price="5",
                recipient="warrior_1",
            ),
            condition=Condition.A,
        )


@pytest.mark.invariant
@pytest.mark.parametrize("recipient", ["warrior_1", "", "none", "None", "NULL"])
def test_condition_a_ignores_the_recipient_of_a_non_message_action(recipient: str) -> None:
    """A stray recipient carries no decision in A, as in B and C: it is dropped."""
    action = parse_action(json_action(action="wait", recipient=recipient), condition=Condition.A)
    assert action.action is ActionType.WAIT
    assert action.recipient is None


@pytest.mark.parametrize("recipient", ["", "none", "None"])
def test_a_blank_recipient_is_null_in_b_and_c(recipient: str) -> None:
    action = parse_action(json_action(action="wait", recipient=recipient), condition=Condition.B)
    assert action.recipient is None
    with pytest.raises(InvalidAgentOutput, match="requires a recipient"):
        parse_action(
            json_action(
                action="send_message",
                recipient=recipient,
                message_type="INFORM",
                target="iron",
            ),
            condition=Condition.B,
        )


def test_prompt_sections_are_labelled_for_bucket_attribution(rig) -> None:
    config, engine, messages = rig
    messages.send(
        sender_id="warrior_1",
        recipient_id="woodworking_1",
        message_type=MessageType.OFFER,
        tick=0,
    )
    observation = build_observation(config, engine, "woodworking_1", messages=messages)
    prompt = build_agent_prompt(observation, config)

    assert SECTION_P2P in prompt.sections
    assert SECTION_BULLETIN not in prompt.sections
    assert prompt.section_sizes()[SECTION_P2P] > 0


@pytest.mark.invariant
def test_the_world_rules_state_what_peers_can_see() -> None:
    """Design s7: the observability rule of s5.2 is world documentation.

    Without it the agent's world model is incomplete in exactly the place that
    governs whether messaging has any point. Descriptive only -- the clause says
    what peers can see, never what to do about it; a tactical form of the same
    idea is inadmissible because it would coach the coordination behaviour the
    experiment measures (IMPLEMENTATION.md section 4.8.1).
    """
    from rpg_economy.agents.prompts import RULES

    assert "Peers cannot see your needs" in RULES
    # The inadmissible clause, in any recognisable form.
    lowered = RULES.lower()
    for coaching in ("order book only helps", "you should message", "ask a peer when"):
        assert coaching not in lowered, f"tactical coaching leaked into the world rules: {coaching}"


@pytest.mark.invariant
def test_every_valid_action_documents_its_parameters() -> None:
    """Every valid action, cancel_order included, has a parameter line.

    cancel_order overloads `quantity` to carry an order id (IMPLEMENTATION.md
    section 4.8.1). An action the schema names but never explains differentially
    punishes whichever model tries it.
    """
    from rpg_economy.agents.prompts import ACTION_SCHEMA
    from rpg_economy.domain import ActionType

    documented = {
        line.split()[1]
        for line in ACTION_SCHEMA.splitlines()
        if line.startswith("- ") and len(line.split()) > 1
    }
    missing = [
        action.value
        for action in ActionType
        if action.value not in documented and action.value not in {"buy", "sell"}
    ]
    assert "buy/sell" in ACTION_SCHEMA, "buy and sell are documented as a pair"
    assert not missing, f"actions named as valid but never explained: {missing}"


# ---------------------------------------------------------------------------
# Peer messages are structured (design s6)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"message_type": "OFFER", "quantity": 1, "price": "5"}, "target"),
        ({"message_type": "OFFER", "target": "gold", "quantity": 1, "price": "5"}, "good"),
        ({"message_type": "REQUEST", "target": "iron", "price": "5"}, "quantity"),
        ({"message_type": "OFFER", "target": "iron", "quantity": 1}, "price"),
        ({"message_type": "TASK", "target": "iron"}, "peer-to-peer"),
        ({"target": "iron"}, "message_type"),
    ],
)
def test_an_incomplete_peer_message_is_invalid(fields, reason) -> None:
    with pytest.raises(InvalidAgentOutput, match=reason):
        parse_action(
            json_action(action="send_message", recipient="warrior_1", **fields),
            condition=Condition.B,
        )


def test_an_inform_needs_only_the_good() -> None:
    action = parse_action(
        json_action(
            action="send_message", message_type="INFORM", target="iron", recipient="warrior_1"
        ),
        condition=Condition.B,
    )
    assert action.message_type is MessageType.INFORM
    assert action.target == "iron"


def test_a_stray_message_type_on_another_action_is_ignored() -> None:
    """Pilot: a message_type left on a cancel_order discarded the whole decision."""
    action = parse_action(json_action(action="wait", message_type="OFFER"), condition=Condition.B)
    assert action.action is ActionType.WAIT
    assert action.message_type is None


@pytest.mark.parametrize(
    "fields",
    [
        {"order_id": 23},
        {"target": 23},
        {"target": "#23"},
        {"quantity": 23},
    ],
)
def test_cancel_order_takes_the_id_where_models_put_it(fields: dict) -> None:
    """Pilot: models wrote order_id, or a bare number in target."""
    action = parse_action(json_action(action="cancel_order", **fields), condition=Condition.B)
    assert action.action is ActionType.CANCEL_ORDER
    assert action.quantity == 23


def test_cancel_order_inside_place_orders_takes_order_id() -> None:
    text = json_action(
        action="place_orders",
        orders=[
            {"action": "cancel_order", "target": None, "order_id": 23},
            {"action": "buy", "target": "herbs", "quantity": 2, "price": 3.2},
        ],
    )
    action = parse_action(text, condition=Condition.B)
    assert action.orders[0].quantity == 23


def test_the_artisan_brief_says_artisans_do_not_go_on_expeditions() -> None:
    from rpg_economy.agents.prompts import ROLE_BRIEF
    from rpg_economy.domain import Role

    assert "do not go on expeditions" in ROLE_BRIEF[Role.ARTISAN]


def test_a_sent_message_logs_its_good_and_terms(rig) -> None:
    config, engine, messages = rig
    agent, _ = runner(
        config,
        engine,
        messages,
        always(
            json_action(
                action="send_message",
                message_type="OFFER",
                target="iron",
                quantity=2,
                price="7",
                recipient="warrior_1",
            )
        ),
    )
    agent.activate(engine, "woodworking_1")
    engine.log.close()
    sent = [e for e in read_events(engine.log.path) if e.type is EventType.MESSAGE_SENT]
    assert sent[-1].payload["good"] == "iron"
    assert sent[-1].payload["quantity"] == 2
    assert sent[-1].payload["type"] == "OFFER"


# ---------------------------------------------------------------------------
# Prompt content per condition (design s3.1, s5.2)
# ---------------------------------------------------------------------------
def test_condition_a_prompts_do_not_document_peer_messages(rig) -> None:
    config, engine, messages = rig
    observation = build_observation(config, engine, "woodworking_1", messages=messages)
    a = build_agent_prompt(observation, config, condition=Condition.A)
    b = build_agent_prompt(observation, config, condition=Condition.B)

    assert "send_message" not in a.user
    assert '"needs"' in a.user and "observations" in a.user
    assert "send_message" in b.user
    assert '"needs"' not in b.user


def test_every_condition_gets_the_same_world_rules(rig) -> None:
    """Recipes and zones are rules of the world, not information one condition earns."""
    config, engine, messages = rig
    observation = build_observation(config, engine, "smithing_1", messages=messages)
    systems = {
        condition: build_agent_prompt(observation, config, condition=condition).system
        for condition in Condition
    }
    assert len(set(systems.values())) == 1
    assert "melee_weapon (smithing)" in systems[Condition.A]


def test_an_agent_sees_its_own_recent_actions(rig) -> None:
    config, engine, messages = rig
    agent, _ = runner(
        config,
        engine,
        messages,
        always(json_action(action="sell", target="iron", quantity=99, price=10.0)),
    )
    agent.activate(engine, "warrior_1")
    observation = build_observation(
        config,
        engine,
        "warrior_1",
        messages=messages,
        history=list(agent._history["warrior_1"]),
    )
    prompt = build_agent_prompt(observation, config)
    assert '"your_recent_actions"' in prompt.sections["state"]
    assert '"accepted":false' in prompt.sections["state"]


def test_nulls_and_unknown_keys_do_not_invalidate_an_action() -> None:
    action = parse_action(
        '{"action": "wait", "priority": null, "short_reason": null, "confidence": 0.9}',
        condition=Condition.B,
    )
    assert action.priority is Priority.MEDIUM
    assert action.short_reason == ""


def test_the_role_briefs_give_no_tactical_advice() -> None:
    from rpg_economy.agents.prompts import ROLE_BRIEF

    assert set(ROLE_BRIEF) == {Role.ADVENTURER, Role.ARTISAN}
    for brief in ROLE_BRIEF.values():
        assert "buy below" not in brief.lower() and "sell above" not in brief.lower()


# ---------------------------------------------------------------------------
# Leniency: what carries no decision is not invalid output
# ---------------------------------------------------------------------------
def test_null_result_lists_read_as_empty() -> None:
    from rpg_economy.agents.parsing import parse_activation

    _, fields = parse_activation(
        json_action(action="wait", needs=None, offers=None, observations=None),
        condition=Condition.A,
    )
    assert fields.needs == [] and fields.offers == [] and fields.observations == []


def test_result_entries_drop_unknown_keys_and_fold_case() -> None:
    from rpg_economy.agents.parsing import parse_activation

    _, fields = parse_activation(
        json_action(
            action="wait",
            needs=[{"good": " IRON ", "quantity": 2, "priority": "HIGH", "why": "forge idle"}],
            offers=[{"good": "Melee_Weapon", "quantity": 1, "price": 30, "note": "x"}],
            observations=[{"signal": "SCARCE", "good": "Iron", "comment": "dear"}],
        ),
        condition=Condition.A,
    )
    assert fields.needs[0].good == "iron"
    assert fields.needs[0].priority is Priority.HIGH
    assert fields.offers[0].good == "melee_weapon"
    assert fields.observations[0].signal.value == "scarce"
    assert fields.observations[0].good == "iron"


def test_normalize_entry_leaves_decisions_to_validation() -> None:
    from rpg_economy.agents.parsing import NEED_FIELDS, normalize_entry

    assert normalize_entry({"good": "gold", "quantity": 0, "x": 1}, NEED_FIELDS) == {
        "good": "gold",
        "quantity": 0,
    }
    assert normalize_entry({"good": "iron", "quantity": 1, "priority": None}, NEED_FIELDS) == {
        "good": "iron",
        "quantity": 1,
    }
    assert normalize_entry("not an entry", NEED_FIELDS) == "not an entry"


def test_an_overlong_short_reason_is_cut_not_refused() -> None:
    from rpg_economy.agents.parsing import SHORT_REASON_LIMIT

    action = parse_action(
        json_action(action="wait", short_reason="x" * (SHORT_REASON_LIMIT + 100)),
        condition=Condition.B,
    )
    assert len(action.short_reason) == SHORT_REASON_LIMIT


def test_cancel_order_accepts_the_order_id_in_target() -> None:
    action = parse_action(json_action(action="cancel_order", target=" 17 "), condition=Condition.B)
    assert action.action is ActionType.CANCEL_ORDER
    assert action.quantity == 17


def test_cancel_order_without_an_id_anywhere_is_invalid() -> None:
    with pytest.raises(InvalidAgentOutput, match="order id"):
        parse_action(json_action(action="cancel_order", target="iron"), condition=Condition.B)


def test_an_empty_message_type_outside_send_message_is_ignored() -> None:
    action = parse_action(json_action(action="wait", message_type=""), condition=Condition.B)
    assert action.message_type is None


# ---------------------------------------------------------------------------
# A reply cut off by the output cap is told apart from a malformed one
# ---------------------------------------------------------------------------
class _TruncatingClient(MockLLMClient):
    def complete(self, *, system: str, user: str):
        import dataclasses

        return dataclasses.replace(
            super().complete(system=system, user=user), finish_reason="length"
        )


def test_a_truncated_reply_is_flagged_as_truncated(rig) -> None:
    config, engine, messages = rig
    client = _TruncatingClient(always('{"action": "wa'))
    agent = LLMAgentRunner(config, client, TokenLedger(), messages=messages)
    agent.activate(engine, "woodworking_1")
    engine.log.close()

    invalid = [e for e in read_events(engine.log.path) if e.type is EventType.AGENT_OUTPUT_INVALID]
    assert invalid
    assert all(e.payload["truncated"] is True for e in invalid)


def test_a_malformed_complete_reply_is_not_flagged_as_truncated(rig) -> None:
    config, engine, messages = rig
    agent, _ = runner(config, engine, messages, always("garbage"))
    agent.activate(engine, "woodworking_1")
    engine.log.close()

    invalid = [e for e in read_events(engine.log.path) if e.type is EventType.AGENT_OUTPUT_INVALID]
    assert invalid
    assert all(e.payload["truncated"] is False for e in invalid)


# ---------------------------------------------------------------------------
# The observation separates what is held from what is still free to use
# ---------------------------------------------------------------------------
def test_the_observation_shows_unreserved_inventory_apart_from_the_total(rig) -> None:
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    account = engine.state.account("woodworking_1")
    account.inventory.clear()
    account.receive(Resource.IRON, 5)
    engine.submit_order("woodworking_1", OrderSide.SELL, Resource.IRON, 3, "500")

    observation = build_observation(config, engine, "woodworking_1", messages=messages)

    assert observation.inventory["iron"] == 5
    assert observation.available_inventory["iron"] == 2
    assert observation.to_payload()["available_inventory"]["iron"] == 2


def test_fully_reserved_stock_is_absent_from_the_available_inventory(rig) -> None:
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    account = engine.state.account("woodworking_1")
    account.inventory.clear()
    account.receive(Resource.IRON, 3)
    engine.submit_order("woodworking_1", OrderSide.SELL, Resource.IRON, 3, "500")

    observation = build_observation(config, engine, "woodworking_1", messages=messages)

    assert observation.inventory == {"iron": 3}
    assert "iron" not in observation.available_inventory


def test_the_observation_shows_unreserved_gold_apart_from_the_total(rig) -> None:
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    account = engine.state.account("woodworking_1")
    total = account.gold
    engine.submit_order("woodworking_1", OrderSide.BUY, Resource.IRON, 2, "10")

    observation = build_observation(config, engine, "woodworking_1", messages=messages)

    assert observation.gold == total
    assert observation.available_gold == total - Decimal("20")


# ---------------------------------------------------------------------------
# place_orders: several market orders in one activation
# ---------------------------------------------------------------------------
def _orders_json(orders: list[dict], **extra) -> str:
    import json

    return json.dumps({"action": "place_orders", "orders": orders, "short_reason": "", **extra})


def test_place_orders_parses_each_order_as_the_action_it_is() -> None:
    action = parse_action(
        _orders_json(
            [
                {"action": "sell", "target": "iron", "quantity": 3, "price": 10.5},
                {"action": "buy", "target": "health_potion", "quantity": 2, "price": 15},
                {"action": "cancel_order", "target": "7"},
            ]
        ),
        condition=Condition.B,
    )
    assert action.action is ActionType.PLACE_ORDERS
    assert [o.action for o in action.orders] == [
        ActionType.SELL,
        ActionType.BUY,
        ActionType.CANCEL_ORDER,
    ]
    assert action.orders[0].price == Decimal("10.5")
    assert action.orders[2].quantity == 7


def test_place_orders_is_capped() -> None:
    order = {"action": "sell", "target": "iron", "quantity": 1, "price": 10}
    parse_action(_orders_json([order] * 4), condition=Condition.B, max_orders=4)
    with pytest.raises(InvalidAgentOutput, match="at most 4"):
        parse_action(_orders_json([order] * 5), condition=Condition.B, max_orders=4)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [{"action": "produce", "target": "armor", "quantity": 1}],
        [{"action": "start_expedition", "target": "green_hollow"}],
        [{"action": "sell", "target": "iron", "quantity": 1}],  # no price
        ["sell iron"],
    ],
)
def test_place_orders_takes_only_complete_market_orders(bad: list) -> None:
    with pytest.raises(InvalidAgentOutput):
        parse_action(_orders_json(bad), condition=Condition.B)


def test_a_stray_orders_list_beside_another_action_is_ignored() -> None:
    """Like a stray message_type: the action named is the one the agent chose."""
    action = parse_action(
        json_action(action="wait")[:-1]
        + ', "orders": [{"action": "sell", "target": "iron", "quantity": 1, "price": 9}]}',
        condition=Condition.B,
    )
    assert action.action is ActionType.WAIT
    assert not action.orders


def test_place_orders_is_available_in_condition_a_too() -> None:
    action = parse_action(
        _orders_json([{"action": "sell", "target": "iron", "quantity": 1, "price": 9}]),
        condition=Condition.A,
    )
    assert action.action is ActionType.PLACE_ORDERS


def test_every_schema_explains_place_orders_and_the_cap_is_in_the_world_rules() -> None:
    from rpg_economy.agents.prompts import ACTION_SCHEMA, ACTION_SCHEMA_A, render_world

    for schema in (ACTION_SCHEMA, ACTION_SCHEMA_A):
        assert "place_orders" in schema and '"orders"' in schema
    config = make_config()
    cap = config.market.max_orders_per_activation
    assert f"one place_orders carries at most {cap} orders" in render_world(config)


def test_the_mock_writes_orders_the_parser_reads_back() -> None:
    from rpg_economy.agents.offline import action_to_json

    orders = (
        AgentAction(action=ActionType.SELL, target="iron", quantity=2, price=Decimal("10.5")),
        AgentAction(action=ActionType.BUY, target="health_potion", quantity=1, price=Decimal("15")),
    )
    action = AgentAction(action=ActionType.PLACE_ORDERS, orders=orders, short_reason="x")
    back = parse_action(action_to_json(action), condition=Condition.B)
    assert [(o.action, o.target, o.quantity, o.price) for o in back.orders] == [
        (o.action, o.target, o.quantity, o.price) for o in orders
    ]


# ---------------------------------------------------------------------------
# Prompt 2.7: the extra rest after a failed trip
# ---------------------------------------------------------------------------
def test_the_failure_rest_clause_states_the_extra_rest_by_risk() -> None:
    from rpg_economy.agents.prompts import render_world

    text = render_world(make_config())
    assert "after a failed one, more by zone risk: high +4, low +1, medium +2 ticks" in text

    none = make_config(
        expeditions={"recovery_ticks_on_failure": {"low": 0, "medium": 0, "high": 0}}
    )
    assert "after a failed one" not in render_world(none)


def test_the_own_order_clause_follows_the_self_trade_switch() -> None:
    from rpg_economy.agents.prompts import render_world

    assert "open order of the same agent" in render_world(make_config())
    allowed = make_config(market={"allow_self_trade": True})
    assert "open order of the same agent" not in render_world(allowed)


# ---------------------------------------------------------------------------
# Prompt 2.2: neutral world rules
# ---------------------------------------------------------------------------
def test_the_rules_say_place_orders_is_judged_order_by_order() -> None:
    from rpg_economy.agents.prompts import RULES

    assert "not partially applied, except place_orders" in RULES


def test_the_adventurer_brief_states_the_away_rule_and_no_fixed_potion_use() -> None:
    from rpg_economy.agents.prompts import ROLE_BRIEF

    brief = ROLE_BRIEF[Role.ADVENTURER]
    assert "each trip uses one up" not in brief
    assert "While away on an expedition you can trade" in brief
    assert "cannot equip or set out" in brief


def test_production_is_described_in_recipe_runs() -> None:
    from rpg_economy.agents.prompts import ACTION_SCHEMA, ACTION_SCHEMA_A, ROLE_BRIEF

    for schema in (ACTION_SCHEMA, ACTION_SCHEMA_A):
        assert "a number of recipe runs" in schema
        assert "production_capacity_left counts runs" in schema
    assert "capacity counted in recipe runs" in ROLE_BRIEF[Role.ARTISAN]


def test_the_peer_visibility_line_fits_each_condition() -> None:
    """Only B and C can tell a peer anything; in A the channel is the RESULT."""
    from rpg_economy.agents.prompts import ACTION_SCHEMA, ACTION_SCHEMA_A, RULES

    assert "unless you tell them" not in RULES
    assert "Peers cannot see your needs unless you tell them" in ACTION_SCHEMA
    assert "unless you tell them" not in ACTION_SCHEMA_A
    assert "You cannot message other agents" in ACTION_SCHEMA_A


def test_the_messaging_text_is_protocol_not_tactics() -> None:
    from rpg_economy.agents.prompts import ACTION_SCHEMA

    lowered = ACTION_SCHEMA.lower()
    for tactic in ("another counterparty", "change your price", "look for"):
        assert tactic not in lowered


def test_the_task_section_states_the_task_purpose(rig) -> None:
    from rpg_economy.domain import TaskType
    from rpg_economy.orchestration.protocol import TASK_PURPOSES

    config, engine, messages = rig
    observation = build_observation(config, engine, "smithing_1", messages=messages)
    task = {"task_id": "task-1", "type": "MANAGE_PROCUREMENT", "params": {}, "context": []}
    prompt = build_agent_prompt(observation, config, condition=Condition.A, task=task)
    assert f"MANAGE_PROCUREMENT: {TASK_PURPOSES[TaskType.MANAGE_PROCUREMENT]}" in prompt.user


# ---------------------------------------------------------------------------
# Parsing: ids are case-folded
# ---------------------------------------------------------------------------
def test_a_capitalized_target_names_the_same_good() -> None:
    upper = parse_action(
        json_action(action="sell", target=" Iron ", quantity=1, price=10.0),
        condition=Condition.B,
    )
    assert upper.target == "iron"

    batch = parse_action(
        '{"action": "place_orders", "orders": ['
        '{"action": "buy", "target": "HEALTH_POTION", "quantity": 1, "price": 5},'
        '{"action": "sell", "target": "Iron", "quantity": 2, "price": 9}]}',
        condition=Condition.B,
    )
    assert [o.target for o in batch.orders] == ["health_potion", "iron"]

    message = parse_action(
        json_action(
            action="send_message",
            message_type="OFFER",
            target="IRON",
            quantity=1,
            price=10.0,
            recipient="smithing_1",
        ),
        condition=Condition.B,
    )
    assert message.target == "iron"


# ---------------------------------------------------------------------------
# The WAIT after invalid output, and its log record
# ---------------------------------------------------------------------------
class FixedCostClient:
    """Returns the same text at a fixed token cost per call."""

    model = "fixed"

    def __init__(self, text: str, tokens: int) -> None:
        self.text, self.tokens, self.calls = text, tokens, 0

    def complete(self, *, system: str, user: str):
        from rpg_economy.llm.client import LLMResponse

        self.calls += 1
        return LLMResponse(
            text=self.text, input_tokens=self.tokens, output_tokens=0, model=self.model
        )


def test_the_wait_says_invalid_output_when_the_retry_spends_the_budget(rig) -> None:
    """Both attempts invalid; the second call exhausts the budget. The loop
    ended because the output was invalid, not because the budget stopped it."""
    from rpg_economy.llm.budget import TokenBudget

    config, engine, messages = rig
    client = FixedCostClient("not json", tokens=100)
    budget = TokenBudget(cap=150)
    agents = LLMAgentRunner(config, client, TokenLedger(), messages=messages, budget=budget)
    result = agents.activate(engine, "woodworking_1")

    assert client.calls == 2
    assert budget.exhausted
    assert result.action.action is ActionType.WAIT
    assert result.action.short_reason == "invalid output after retry"


def test_the_wait_says_invalid_output_when_the_budget_stops_the_retry(rig) -> None:
    """The first attempt was invalid and spent the budget, so no retry was
    made. A call was made and its output was invalid: that is why the
    activation ended in WAIT, not that the budget stopped it."""
    from rpg_economy.agents.llm_agent import WAIT_INVALID_NO_RETRY
    from rpg_economy.llm.budget import TokenBudget

    config, engine, messages = rig
    client = FixedCostClient("not json", tokens=100)
    budget = TokenBudget(cap=50)
    agents = LLMAgentRunner(config, client, TokenLedger(), messages=messages, budget=budget)
    result = agents.activate(engine, "woodworking_1")

    assert client.calls == 1
    assert result.action.short_reason == WAIT_INVALID_NO_RETRY
    assert "invalid output" in result.action.short_reason
    assert result.invalid_outputs == 1


def test_the_wait_says_budget_exhausted_only_when_no_call_was_made(rig) -> None:
    from rpg_economy.llm.budget import TokenBudget

    config, engine, messages = rig
    client = FixedCostClient("not json", tokens=100)
    budget = TokenBudget(cap=50)
    budget.charge(60, tick=0, cycle=0)
    assert not budget.may_call()
    agents = LLMAgentRunner(config, client, TokenLedger(), messages=messages, budget=budget)
    result = agents.activate(engine, "woodworking_1")

    assert client.calls == 0
    assert result.action.short_reason == "budget exhausted"


def test_invalid_output_is_logged_in_full(rig) -> None:
    config, engine, messages = rig
    long_invalid = "no action here " * 60
    agents, _ = runner(config, engine, messages, always(long_invalid))
    agents.activate(engine, "woodworking_1")
    engine.log.close()

    invalid = [e for e in read_events(engine.log.path) if e.type is EventType.AGENT_OUTPUT_INVALID]
    assert len(invalid) == MAX_ATTEMPTS
    assert all(e.payload["raw"] == long_invalid for e in invalid)


# ---------------------------------------------------------------------------
# The offline mock: its messaging draws leave the agents' decisions alone
# ---------------------------------------------------------------------------
def test_the_mock_draws_its_messages_from_a_stream_of_its_own(tmp_path: Path) -> None:
    """B's mock messages must not shift the rule-based draws A makes.

    The same artisans answer the same prompts in A and in B; afterwards every
    agent's own stream is in the same state in both, while B did send messages.
    """
    import json as _json

    from rpg_economy.agents.offline import rule_based_responder

    states = {}
    sent = {}
    for condition in ("A", "B"):
        config = make_config(run={"condition": condition})
        with EventLogWriter(tmp_path / f"{condition}.jsonl", make_metadata(config)) as log:
            engine = Engine(config, build_world(config), log)
            # A smith with only a finished weapon to sell and no gold: its
            # action is a single sell, which the mock may turn into an OFFER.
            smith = engine.state.account("smithing_1")
            smith.inventory.clear()
            smith.gold = Decimal(0)
            smith.receive(Product.MELEE_WEAPON, 3)
            respond = rule_based_responder(config, engine, reply_to_messages=False)
            replies = [_json.loads(respond("", '"agent_id":"smithing_1"')) for _ in range(20)]
            sent[condition] = sum(1 for r in replies if r["action"] == "send_message")
            states[condition] = {
                a: engine.random.agent(a).getstate() for a in engine.state.agent_ids
            }

    assert sent["A"] == 0
    assert sent["B"] > 0, "the mock sent no message; the test proves nothing"
    assert states["A"] == states["B"]


def test_the_mock_never_trades_a_batch_for_one_message() -> None:
    from rpg_economy.agents.offline import _single_quote

    sell = AgentAction(action=ActionType.SELL, target="iron", quantity=1, price=Decimal("9"))
    buy = AgentAction(action=ActionType.BUY, target="wood", quantity=1, price=Decimal("5"))
    assert _single_quote(sell) is sell
    one = AgentAction(action=ActionType.PLACE_ORDERS, orders=(buy,))
    assert _single_quote(one) is buy
    batch = AgentAction(action=ActionType.PLACE_ORDERS, orders=(sell, buy))
    assert _single_quote(batch) is None
    assert _single_quote(AgentAction(action=ActionType.WAIT)) is None


# ---------------------------------------------------------------------------
# Prompt 2.3: available gold, the wealth tax, one open proposal per good
# ---------------------------------------------------------------------------
def test_the_rules_say_what_available_gold_leaves_out() -> None:
    from rpg_economy.agents.prompts import RULES

    assert "gold your open orders reserve, frozen gold, and wealth tax you owe" in RULES


def test_the_costs_state_the_wealth_tax_as_the_engine_charges_it() -> None:
    from rpg_economy.agents.prompts import render_world

    world = render_world(make_config())
    assert "frozen gold and tax already owed not taxed" in world
    assert "is owed (tax_owed), collected from free gold at the start of every tick" in world


def test_the_message_rules_state_one_open_proposal_per_good() -> None:
    from rpg_economy.agents.prompts import ACTION_SCHEMA, ACTION_SCHEMA_A

    rule = "At most one REQUEST or OFFER from one agent to another about a good is open"
    assert rule in ACTION_SCHEMA
    assert rule not in ACTION_SCHEMA_A


def test_the_adventurer_brief_claims_no_spread_the_config_lacks() -> None:
    """yield_spread is equal for medium and high risk, so no 'wider spread'."""
    from rpg_economy.agents.prompts import ROLE_BRIEF
    from rpg_economy.domain import Role

    assert "spread" not in ROLE_BRIEF[Role.ADVENTURER]


# ---------------------------------------------------------------------------
# RESULT fields: Condition A only
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("price", [0, "0", 0.0])
def test_an_offer_priced_at_zero_is_read_as_unpriced(price) -> None:
    from rpg_economy.agents.parsing import parse_activation

    text = json_action(action="wait", offers=[{"good": "iron", "quantity": 2, "price": price}])
    _, fields = parse_activation(text, condition=Condition.A)
    (offer,) = fields.offers
    assert offer.price is None


def test_an_offer_with_a_negative_price_is_dropped_not_the_action() -> None:
    from rpg_economy.agents.parsing import parse_activation

    text = json_action(action="wait", offers=[{"good": "iron", "quantity": 2, "price": -1}])
    action, fields = parse_activation(text, condition=Condition.A)
    assert action.action is ActionType.WAIT
    assert fields.offers == []


@pytest.mark.parametrize(
    "fields",
    [
        {"needs": "iron please"},
        {"offers": [{"good": "gold", "quantity": 1}]},
        {"observations": [{"signal": "x", "good": "iron"}] * 11},
    ],
)
def test_b_and_c_ignore_result_fields_that_a_would_refuse(fields) -> None:
    from rpg_economy.agents.parsing import parse_activation

    text = json_action(action="wait", **fields)
    for condition in (Condition.B, Condition.C):
        action, result = parse_activation(text, condition=condition)
        assert action.action is ActionType.WAIT
        assert result.needs == result.offers == result.observations == []
    # In A the malformed entries are dropped and the decision stands.
    action, result = parse_activation(text, condition=Condition.A)
    assert action.action is ActionType.WAIT


def test_a_keeps_at_most_the_cap_of_result_entries() -> None:
    from rpg_economy.agents.parsing import MAX_RESULT_ENTRIES, parse_activation

    needs = [{"good": "iron", "quantity": 1}] * (MAX_RESULT_ENTRIES + 1)
    _, fields = parse_activation(json_action(action="wait", needs=needs), condition=Condition.A)
    assert len(fields.needs) == MAX_RESULT_ENTRIES


def test_a_malformed_result_entry_is_dropped_and_the_rest_kept() -> None:
    """Pilot: a quantity sent as text discarded whole decisions in A."""
    from rpg_economy.agents.parsing import parse_activation

    text = json_action(
        action="buy",
        target="iron",
        quantity=1,
        price=9,
        needs=[{"good": "iron", "quantity": "a few"}, {"good": "leather", "quantity": 2}],
        observations=[{"signal": "panic", "good": "iron"}],
    )
    action, fields = parse_activation(text, condition=Condition.A)
    assert action.action is ActionType.BUY
    assert [n.good for n in fields.needs] == ["leather"]
    assert fields.observations == []


# ---------------------------------------------------------------------------
# The log: raw completions, delivered messages, the bulletin in a prompt
# ---------------------------------------------------------------------------
def test_a_valid_completion_is_logged_in_full(rig) -> None:
    config, engine, messages = rig
    text = json_action(action="wait", short_reason="nothing to do")
    agent, _ = runner(config, engine, messages, always(text))
    agent.activate(engine, "woodworking_1")
    engine.log.close()

    (call,) = [e for e in read_events(engine.log.path) if e.type is EventType.LLM_CALL]
    assert call.payload["raw"] == text


def test_delivered_messages_are_logged_with_their_tokens(rig) -> None:
    config, engine, messages = rig
    sent = messages.send(
        sender_id="warrior_1",
        recipient_id="woodworking_1",
        message_type=MessageType.INFORM,
        tick=0,
        payload={"good": "iron"},
    )
    agent, ledger = runner(config, engine, messages, always(json_action(action="wait")))
    agent.activate(engine, "warrior_1")  # an empty inbox delivers nothing
    agent.activate(engine, "woodworking_1")
    engine.log.close()

    events = list(read_events(engine.log.path))
    (delivered,) = [e for e in events if e.type is EventType.MESSAGE_DELIVERED]
    assert delivered.agent_id == "woodworking_1"
    assert delivered.payload["message_ids"] == [sent.message_id]
    assert delivered.payload["tokens"] > 0
    assert delivered.payload["tokens"] == ledger.input_tokens(TokenBucket.P2P_CONTENT)
    assert not [e for e in events if e.type is EventType.BULLETIN_INCLUDED_IN_PROMPT]


def test_the_bulletin_is_logged_only_when_a_prompt_carries_it(tmp_path: Path) -> None:
    included = {}
    for condition in (Condition.B, Condition.C):
        config = make_config(run={"condition": condition.value})
        with EventLogWriter(tmp_path / f"{condition.value}.jsonl", make_metadata(config)) as log:
            engine = Engine(config, build_world(config), log)
            engine.publish_bulletin(build_global_summary(config, engine))
            messages = MessageBus(build_directory(engine))
            agent, ledger = runner(config, engine, messages, always(json_action(action="wait")))
            agent.activate(engine, "woodworking_1")
        included[condition] = (
            [e for e in read_events(log.path) if e.type is EventType.BULLETIN_INCLUDED_IN_PROMPT],
            ledger,
        )

    assert included[Condition.B][0] == []
    (event,), ledger = included[Condition.C]
    assert event.agent_id == "woodworking_1"
    assert event.payload["tokens"] == ledger.input_tokens(TokenBucket.BULLETIN) > 0


# ---------------------------------------------------------------------------
# Parsing parity with the Director's tolerance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spelling", ["start_expedition", "start-expedition", "Start Expedition", " START-EXPEDITION "]
)
def test_an_action_spelled_with_hyphens_or_spaces_is_read(spelling: str) -> None:
    action = parse_action(
        json_action(action=spelling, target="green_hollow"), condition=Condition.B
    )
    assert action.action is ActionType.START_EXPEDITION


def test_a_misspelled_action_is_still_invalid() -> None:
    with pytest.raises(InvalidAgentOutput, match="unknown action"):
        parse_action(json_action(action="start expeditions"), condition=Condition.B)


@pytest.mark.parametrize("message_type", ["", "  ", "none", "None", "null", "NULL"])
@pytest.mark.parametrize("condition", [Condition.A, Condition.B, Condition.C])
def test_a_null_spelled_message_type_outside_send_message_is_ignored(
    message_type: str, condition: Condition
) -> None:
    action = parse_action(
        json_action(action="wait", message_type=message_type), condition=condition
    )
    assert action.message_type is None


def test_a_null_spelled_message_type_on_send_message_is_still_invalid() -> None:
    with pytest.raises(InvalidAgentOutput, match="unknown message_type"):
        parse_action(
            json_action(
                action="send_message", message_type="none", target="iron", recipient="warrior_1"
            ),
            condition=Condition.B,
        )


# ---------------------------------------------------------------------------
# Tax owed is a lien the agent can see (user ruling)
# ---------------------------------------------------------------------------
def test_the_observation_and_prompt_show_the_tax_owed(rig) -> None:
    import json as json_module

    config, engine, messages = rig
    engine.state.account("woodworking_1").tax_arrears = Decimal("12.34")
    observation = build_observation(config, engine, "woodworking_1", messages=messages)
    assert observation.tax_owed == Decimal("12.34")
    assert observation.to_payload()["tax_owed"] == "12.34"

    state = build_agent_prompt(observation, config).sections["state"]
    payload = json_module.loads(state.split("\n", 1)[1])
    assert payload["tax_owed"] == "12.34"


def test_the_tax_owed_is_zero_when_nothing_is_owed(rig) -> None:
    config, engine, messages = rig
    observation = build_observation(config, engine, "woodworking_1", messages=messages)
    assert observation.tax_owed == 0
    assert '"tax_owed":"0' in build_agent_prompt(observation, config).sections["state"]


def test_the_rules_say_owed_tax_is_a_lien_on_available_gold() -> None:
    from rpg_economy.agents.prompts import RULES, render_world

    assert "wealth tax you owe (tax_owed)" in RULES
    assert "Tax owed is collected from free gold (see Costs)" in RULES
    assert "cannot be spent or bid" in RULES
    costs = render_world(make_config())
    assert "tax already owed not taxed" in costs
    assert "cannot be spent or bid meanwhile" in costs


# ---------------------------------------------------------------------------
# Prompt hashes cover every piece of code that shapes prompt text
# ---------------------------------------------------------------------------
def test_the_agent_prompt_hash_covers_every_payload_builder() -> None:
    from rpg_economy.agents.observation import MarketView, Observation
    from rpg_economy.agents.prompts import agent_prompt_sources
    from rpg_economy.messaging import Message
    from rpg_economy.orchestration.protocol import Result, ResultReference, Task

    sources = agent_prompt_sources()
    for function in (
        build_agent_prompt,
        Task.to_payload,
        Task.to_agent_payload,
        ResultReference.public_view,
        Result.to_reference,
        Message.to_payload,
        Observation.message_payloads,
        MarketView.to_payload,
    ):
        assert function in sources
    import inspect

    assert '"tax_owed": observation.tax_owed' in inspect.getsource(build_agent_prompt)
