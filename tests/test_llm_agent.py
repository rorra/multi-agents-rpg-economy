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


def test_the_director_and_global_view_hashes_cover_their_helpers() -> None:
    from rpg_economy.orchestration import global_view
    from rpg_economy.orchestration.director import (
        Director,
        director_prompt_sources,
        global_view_sources,
    )
    from rpg_economy.orchestration.protocol import Result

    assert Result.to_reference in director_prompt_sources()
    assert Director._user_prompt in director_prompt_sources()
    view = global_view_sources()
    assert global_view._mean in view and global_view._stdev in view
    assert global_view.render_global_summary in view


def test_a_change_to_a_payload_builder_changes_the_agent_prompt_hash(monkeypatch) -> None:
    import inspect

    from rpg_economy.messaging import Message

    prompt_hashes.cache_clear()
    before = prompt_hashes()["agent_prompt_builder"]
    real = inspect.getsource

    def edited(obj):
        text = real(obj)
        return text + "# edited\n" if obj is Message.to_payload else text

    monkeypatch.setattr(inspect, "getsource", edited)
    try:
        # Cached: the edit is invisible until the cache is cleared, as a file
        # edited on disk mid-campaign must be.
        assert prompt_hashes()["agent_prompt_builder"] == before
        prompt_hashes.cache_clear()
        assert prompt_hashes()["agent_prompt_builder"] != before
    finally:
        monkeypatch.undo()
        prompt_hashes.cache_clear()


# ---------------------------------------------------------------------------
# Delivery events repeat per attempt and name their activation
# ---------------------------------------------------------------------------
def test_a_retry_repeats_delivery_events_under_one_activation(tmp_path: Path) -> None:
    config = make_config(run={"condition": "C"})
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        engine.publish_bulletin(build_global_summary(config, engine))
        messages = MessageBus(build_directory(engine))
        messages.send(
            sender_id="warrior_1",
            recipient_id="woodworking_1",
            message_type=MessageType.INFORM,
            tick=0,
            payload={"good": "iron"},
        )
        agent, ledger = runner(
            config,
            engine,
            messages,
            scripted(["not json", json_action(action="wait"), json_action(action="wait")]),
        )
        agent.activate(engine, "woodworking_1")  # invalid, then the retry
        agent.activate(engine, "woodworking_1")  # a second activation, same tick
    events = list(read_events(log.path))

    for kind, bucket in (
        (EventType.MESSAGE_DELIVERED, TokenBucket.P2P_CONTENT),
        (EventType.BULLETIN_INCLUDED_IN_PROMPT, TokenBucket.BULLETIN),
    ):
        delivered = [e for e in events if e.type is kind]
        assert [e.payload["attempt"] for e in delivered] == [1, 2, 1]
        activations = [e.payload["activation"] for e in delivered]
        assert activations[0] == activations[1] != activations[2]
        assert activations[0] == {"tick": engine.tick, "agent_id": "woodworking_1", "seq": 1}
        assert activations[2]["seq"] == 2
        # Tokens are per call and sum to the bucket; deliveries de-duplicate.
        assert sum(e.payload["tokens"] for e in delivered) == ledger.input_tokens(bucket)
        assert len({tuple(sorted(a.items())) for a in activations}) == 2


# ---------------------------------------------------------------------------
# Still-open proposals are shown until answered or expired (B and C)
# ---------------------------------------------------------------------------
def _recording():
    """A responder that records each user prompt and answers WAIT."""
    prompts: list[str] = []

    def respond(system: str, user: str) -> str:
        prompts.append(user)
        return json_action(action="wait")

    return prompts, respond


def _offer(messages: MessageBus, tick: int = 0, good: str = "iron") -> None:
    messages.send(
        sender_id="warrior_1",
        recipient_id="woodworking_1",
        message_type=MessageType.OFFER,
        tick=tick,
        payload={"good": good, "quantity": 3, "price": "7.50"},
    )


@pytest.fixture
def proposal_rig(tmp_path: Path):
    """Build a rig whose bus expires proposals as a run's does."""
    logs: list[EventLogWriter] = []

    def build(condition: str = "B"):
        config = make_config(run={"condition": condition})
        log = EventLogWriter(tmp_path / f"{condition}.jsonl", make_metadata(config))
        log.__enter__()
        logs.append(log)
        engine = Engine(config, build_world(config), log)
        messages = MessageBus(build_directory(engine))
        messages.set_proposal_expiry(config.market.order_expiry_ticks, lambda: engine.tick)
        return config, engine, messages, log

    yield build
    for log in logs:
        log.close()


def test_a_proposal_is_shown_once_as_new_then_as_still_open(proposal_rig) -> None:
    config, engine, messages, log = proposal_rig()
    expiry = config.market.order_expiry_ticks
    _offer(messages, tick=0)
    prompts, respond = _recording()
    agent, ledger = runner(config, engine, messages, respond)

    agent.activate(engine, "woodworking_1")
    messages.drain("woodworking_1")  # what the trigger layer does after activation
    engine.state.tick = 2
    agent.activate(engine, "woodworking_1")
    log.close()

    first, second = prompts
    # New: under the inbox only, never duplicated as an open proposal.
    assert "Messages addressed to you" in first
    assert "Proposals still open" not in first
    # Drained but unanswered: still visible, with what it takes to answer it.
    assert "Messages addressed to you" not in second
    assert "Proposals still open" in second
    observations = [
        e.payload["observation"]
        for e in read_events(log.path)
        if e.type is EventType.OBSERVATION_DELIVERED
    ]
    assert observations[0]["open_proposals"] == []
    assert observations[1]["open_proposals"] == [
        {
            "message_id": 1,
            "sender_id": "warrior_1",
            "type": "OFFER",
            "good": "iron",
            "quantity": 3,
            "price": "7.50",
            "ticks_left": expiry - 2,
        }
    ]
    assert f'"ticks_left":{expiry - 2}' in second


def test_open_proposals_are_charged_to_the_p2p_bucket(proposal_rig) -> None:
    config, engine, messages, log = proposal_rig()
    _offer(messages, tick=0)
    messages.drain("woodworking_1")
    engine.state.tick = 1
    prompts, respond = _recording()
    agent, ledger = runner(config, engine, messages, respond)
    agent.activate(engine, "woodworking_1")
    log.close()

    assert ledger.input_tokens(TokenBucket.P2P_CONTENT) > 0
    delivered = [e for e in read_events(log.path) if e.type is EventType.MESSAGE_DELIVERED]
    assert len(delivered) == 1
    assert delivered[0].payload["message_ids"] == []
    assert delivered[0].payload["open_proposal_ids"] == [1]
    assert delivered[0].payload["tokens"] == ledger.input_tokens(TokenBucket.P2P_CONTENT)


def test_an_expired_proposal_is_no_longer_shown(proposal_rig) -> None:
    config, engine, messages, log = proposal_rig()
    _offer(messages, tick=0)
    messages.drain("woodworking_1")
    engine.state.tick = config.market.order_expiry_ticks
    prompts, respond = _recording()
    agent, ledger = runner(config, engine, messages, respond)
    agent.activate(engine, "woodworking_1")
    log.close()

    assert "Proposals still open" not in prompts[0]
    assert ledger.input_tokens(TokenBucket.P2P_CONTENT) == 0
    assert not [e for e in read_events(log.path) if e.type is EventType.MESSAGE_DELIVERED]


def test_an_answered_proposal_is_no_longer_shown(proposal_rig) -> None:
    config, engine, messages, log = proposal_rig()
    _offer(messages, tick=0)
    _offer(messages, tick=0, good="wood")
    messages.drain("woodworking_1")
    messages.send(
        sender_id="woodworking_1",
        recipient_id="warrior_1",
        message_type=MessageType.REJECT,
        tick=1,
        payload={"good": "iron"},
    )
    engine.state.tick = 1
    prompts, respond = _recording()
    agent, ledger = runner(config, engine, messages, respond)
    agent.activate(engine, "woodworking_1")
    log.close()

    (observation,) = [
        e.payload["observation"]
        for e in read_events(log.path)
        if e.type is EventType.OBSERVATION_DELIVERED
    ]
    assert [p["good"] for p in observation["open_proposals"]] == ["wood"]


def test_condition_a_shows_no_open_proposals(proposal_rig) -> None:
    config, engine, messages, log = proposal_rig("A")
    _offer(messages, tick=0)
    messages.drain("woodworking_1")
    prompts, respond = _recording()
    agent, ledger = runner(config, engine, messages, respond)
    agent.activate(engine, "woodworking_1")
    log.close()

    assert "Proposals still open" not in prompts[0]
    assert ledger.input_tokens(TokenBucket.P2P_CONTENT) == 0
    (observation,) = [
        e.payload["observation"]
        for e in read_events(log.path)
        if e.type is EventType.OBSERVATION_DELIVERED
    ]
    assert observation["open_proposals"] == []


def test_the_message_rules_say_open_proposals_are_shown_again() -> None:
    from rpg_economy.agents.prompts import ACTION_SCHEMA, ACTION_SCHEMA_A

    assert "stays listed after that, among the open proposals" in ACTION_SCHEMA
    assert "open proposals" not in ACTION_SCHEMA_A
    # The headers appear only where there is content under them.
    assert "Proposals still open" not in ACTION_SCHEMA
    assert "Messages addressed to you" not in ACTION_SCHEMA


# ---------------------------------------------------------------------------
# One activation's events share a join key
# ---------------------------------------------------------------------------
def test_observation_calls_and_deliveries_carry_the_activation_key(tmp_path: Path) -> None:
    config = make_config(run={"condition": "C"})
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        engine.publish_bulletin(build_global_summary(config, engine))
        messages = MessageBus(build_directory(engine))
        messages.send(
            sender_id="warrior_1",
            recipient_id="woodworking_1",
            message_type=MessageType.INFORM,
            tick=0,
            payload={"good": "iron"},
        )
        agent, _ = runner(
            config, engine, messages, scripted(["not json", json_action(action="wait")])
        )
        agent.activate(engine, "woodworking_1")
    events = list(read_events(log.path))
    key = {"tick": engine.tick, "agent_id": "woodworking_1", "seq": 1}
    for kind, count in (
        (EventType.OBSERVATION_DELIVERED, 1),
        (EventType.LLM_CALL, 2),
        (EventType.MESSAGE_DELIVERED, 2),
        (EventType.BULLETIN_INCLUDED_IN_PROMPT, 2),
    ):
        found = [e for e in events if e.type is kind]
        assert len(found) == count, kind
        assert all(e.payload["activation"] == key for e in found), kind


# ---------------------------------------------------------------------------
# Id-like values fold spaces and hyphens, as the Director's params do
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("Arcane Crystal", "arcane_crystal"),
        ("arcane-crystal", "arcane_crystal"),
        (" Melee  Weapon ", "melee_weapon"),
        ("Iron-Marches", "iron_marches"),
    ],
)
def test_an_id_like_target_folds_spaces_and_hyphens(target, expected) -> None:
    action = parse_action(
        json_action(action="buy", target=target, quantity=1, price=5), condition=Condition.B
    )
    assert action.target == expected


def test_orders_and_result_fields_fold_ids_too() -> None:
    from rpg_economy.agents.parsing import parse_activation

    text = (
        '{"action": "place_orders", "orders": [{"action": "sell", "target": "Arcane Crystal", '
        '"quantity": 1, "price": 9}], "needs": [{"good": "health potion", "quantity": 2}], '
        '"observations": [{"signal": "Price-Rising", "good": "arcane-crystal"}]}'
    )
    action, fields = parse_activation(text, condition=Condition.A)
    assert action.orders[0].target == "arcane_crystal"
    assert fields.needs[0].good == "health_potion"
    assert fields.observations[0].signal.value == "price_rising"
    assert fields.observations[0].good == "arcane_crystal"


@pytest.mark.parametrize("responses", [["not json", "still not json"], ["not json", None]])
def test_invalid_outputs_the_action_and_the_waste_carry_the_activation_key(
    tmp_path: Path, responses: list
) -> None:
    """agent_output_invalid, agent_output and activation_wasted join on the key too."""
    from rpg_economy.llm import LLMError

    config = make_config(run={"condition": "B"})
    replies = iter(responses)

    def respond(system: str, user: str) -> str:
        reply = next(replies)
        if reply is None:
            raise LLMError("no completion")  # the llm-error path of an invalid output
        return reply

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        messages = MessageBus(build_directory(engine))
        agent, _ = runner(config, engine, messages, respond)
        result = agent.activate(engine, "woodworking_1")
    assert result.wasted
    events = list(read_events(log.path))
    key = {"tick": engine.tick, "agent_id": "woodworking_1", "seq": 1}
    for kind, count in (
        (EventType.AGENT_OUTPUT_INVALID, 2),
        (EventType.AGENT_OUTPUT, 1),
        (EventType.ACTIVATION_WASTED, 1),
    ):
        found = [e for e in events if e.type is kind]
        assert len(found) == count, kind
        assert all(e.payload["activation"] == key for e in found), kind


def test_a_failed_call_logs_the_prompt_it_attempted(tmp_path: Path) -> None:
    """No reply came back, but what was sent is in the log, and nothing was charged."""
    from rpg_economy.llm import LLMError

    config = make_config(run={"condition": "B"})
    sent: list[tuple[str, str]] = []
    replies = iter([None, json_action(action="wait")])

    def respond(system: str, user: str) -> str:
        sent.append((system, user))
        reply = next(replies)
        if reply is None:
            raise LLMError("no completion")
        return reply

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        messages = MessageBus(build_directory(engine))
        agent, _ = runner(config, engine, messages, respond)
        agent.activate(engine, "woodworking_1")
    events = list(read_events(log.path))
    systems = {
        e.payload["sha"]: e.payload["text"] for e in events if e.type is EventType.PROMPT_SYSTEM
    }
    (failed,) = [e for e in events if e.type is EventType.AGENT_OUTPUT_INVALID]
    (call,) = [e for e in events if e.type is EventType.LLM_CALL]
    assert (systems[failed.payload["system_sha"]], failed.payload["user"]) == sent[0]
    assert (systems[call.payload["system_sha"]], call.payload["user"]) == sent[1]
    assert len(systems) == 1, "the system prompt is logged once, not per attempt"
    assert "input_tokens" not in failed.payload


# ---------------------------------------------------------------------------
# Each call says whether it was billed, as the client reports it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("billed", [False, True])
def test_an_llm_call_records_whether_it_was_billed(rig, billed: bool) -> None:
    from dataclasses import replace

    config, engine, messages = rig

    class Client(MockLLMClient):
        def complete(self, *, system: str, user: str):
            return replace(super().complete(system=system, user=user), billed=billed)

    agent = LLMAgentRunner(
        config, Client(always(json_action(action="wait"))), TokenLedger(), messages=messages
    )
    agent.activate(engine, "woodworking_1")
    engine.log.close()
    (call,) = [e for e in read_events(engine.log.path) if e.type is EventType.LLM_CALL]
    assert call.payload["billed"] is billed


# ---------------------------------------------------------------------------
# Inbox proposals say whether they can still be answered
# ---------------------------------------------------------------------------
def _inbox_of(log_path: Path) -> list[dict]:
    (observation,) = [
        e.payload["observation"]
        for e in read_events(log_path)
        if e.type is EventType.OBSERVATION_DELIVERED
    ]
    return observation["messages"]


def test_a_replaced_inbox_proposal_is_marked_closed(proposal_rig) -> None:
    config, engine, messages, log = proposal_rig()
    expiry = config.market.order_expiry_ticks
    _offer(messages, tick=0)
    _offer(messages, tick=1)  # replaces the first: same sender, same good
    engine.state.tick = 1
    prompts, respond = _recording()
    agent, _ = runner(config, engine, messages, respond)
    agent.activate(engine, "woodworking_1")
    log.close()

    first, second = _inbox_of(log.path)
    assert (first["open"], first["ticks_left"]) == (False, None)
    assert (second["open"], second["ticks_left"]) == (True, expiry)
    # Both are still shown, the closed one as information.
    assert '"message_id":1' in prompts[0] and '"open":false' in prompts[0]
    assert "shown for information only" in prompts[0]


def test_an_expired_inbox_proposal_is_marked_closed(proposal_rig) -> None:
    config, engine, messages, log = proposal_rig()
    _offer(messages, tick=0)
    messages.send(
        sender_id="warrior_1",
        recipient_id="woodworking_1",
        message_type=MessageType.INFORM,
        tick=0,
        payload={"good": "iron"},
    )
    engine.state.tick = config.market.order_expiry_ticks  # never activated meanwhile
    prompts, respond = _recording()
    agent, _ = runner(config, engine, messages, respond)
    agent.activate(engine, "woodworking_1")
    log.close()

    offer, inform = _inbox_of(log.path)
    assert (offer["open"], offer["ticks_left"]) == (False, None)
    # Only REQUESTs and OFFERs can be answered, so only they are marked.
    assert "open" not in inform and "ticks_left" not in inform


def test_the_mock_skips_an_inbox_proposal_marked_closed(rig) -> None:
    from rpg_economy.agents.offline import _reply_to_sender

    config, engine, messages = rig
    closed = {
        "sender_id": "warrior_1",
        "type": "OFFER",
        "sent_tick": engine.tick,
        "payload": {"good": "iron", "quantity": 1, "price": "1.00"},
        "open": False,
    }
    assert _reply_to_sender(engine, "woodworking_1", [closed]) is None
    assert _reply_to_sender(engine, "woodworking_1", [{**closed, "open": True}]) is not None


def test_the_mock_answers_a_still_open_proposal(proposal_rig) -> None:
    """Offline B and C exercise the open-proposals path end to end."""
    from rpg_economy.agents.offline import rule_based_responder

    config, engine, messages, log = proposal_rig()
    # Priced far above anything recent: the mock declines, which always sends.
    messages.send(
        sender_id="warrior_1",
        recipient_id="woodworking_1",
        message_type=MessageType.OFFER,
        tick=0,
        payload={"good": "iron", "quantity": 3, "price": "9999.00"},
    )
    messages.drain("woodworking_1")
    engine.state.tick = 1
    agent, _ = runner(
        config, engine, messages, rule_based_responder(config, engine, messages=messages)
    )
    agent.activate(engine, "woodworking_1")
    log.close()

    events = list(read_events(log.path))
    (output,) = [e.payload for e in events if e.type is EventType.AGENT_OUTPUT]
    assert output["action"] == ActionType.SEND_MESSAGE.value
    assert output["message_type"] == MessageType.REJECT.value
    assert (output["recipient"], output["target"]) == ("warrior_1", "iron")
    assert messages.open_proposals("woodworking_1", tick=1) == []


# ---------------------------------------------------------------------------
# Equipment wear is stated from the configuration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (1, "each expedition takes 1 durability point from every equipped item"),
        (3, "each expedition takes 3 durability points from every equipped item"),
        (0, "Equipment wear: none"),
    ],
)
def test_the_world_states_the_wear_per_expedition(loss: int, expected: str) -> None:
    from rpg_economy.agents.prompts import ROLE_BRIEF, render_world

    config = make_config(expeditions={"durability_loss_per_expedition": loss})
    assert expected in render_world(config)
    # The brief no longer equates durability with expeditions left.
    assert "expeditions each has left" not in ROLE_BRIEF[Role.ADVENTURER]


def test_an_agent_sees_its_own_recent_trades(tmp_path: Path) -> None:
    """Design s5.2: an agent observes its own orders and trades."""
    from decimal import Decimal

    from rpg_economy.agents.observation import build_observation
    from rpg_economy.domain import OrderSide, Resource

    config = make_config(run={"condition": "B"})
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        seller, buyer = "warrior_1", "smithing_1"
        engine.state.account(seller).receive(Resource.IRON, 2)
        engine.submit_order(seller, OrderSide.SELL, Resource.IRON, 2, Decimal("9"))
        engine.submit_order(buyer, OrderSide.BUY, Resource.IRON, 2, Decimal("9"))
        seen = build_observation(config, engine, buyer)
    assert seen.own_trades == [
        {"tick": engine.tick, "side": "buy", "good": "iron", "quantity": 2, "price": "9"}
    ]


def test_the_logged_observation_carries_the_own_trades(tmp_path: Path) -> None:
    """observation_delivered is the full observation (s15): own trades included."""
    from decimal import Decimal

    from rpg_economy.agents.observation import build_observation
    from rpg_economy.domain import OrderSide, Resource

    config = make_config(run={"condition": "B"})
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        engine.state.account("warrior_1").receive(Resource.IRON, 1)
        engine.submit_order("warrior_1", OrderSide.SELL, Resource.IRON, 1, Decimal("9"))
        engine.submit_order("smithing_1", OrderSide.BUY, Resource.IRON, 1, Decimal("9"))
        payload = build_observation(config, engine, "warrior_1").to_payload()
    assert payload["own_trades"] == [
        {"tick": engine.tick, "side": "sell", "good": "iron", "quantity": 1, "price": "9"}
    ]


def test_prose_after_the_json_object_does_not_spoil_it() -> None:
    from rpg_economy.agents.parsing import extract_json

    text = 'Here: {"action": "wait"} and a note with a brace } at the end.'
    assert extract_json(text) == {"action": "wait"}


def test_a_reply_cut_off_mid_batch_is_invalid_not_its_first_order() -> None:
    """An inner object of an unfinished reply must never pass for the decision."""
    from rpg_economy.agents.parsing import InvalidAgentOutput, parse_activation
    from rpg_economy.domain import Condition

    cut = (
        '{"action": "place_orders", "orders": [{"action": "buy", "target": "iron", '
        '"quantity": 2, "price": 9}, {"action": "sell", "tar'
    )
    with pytest.raises(InvalidAgentOutput):
        parse_activation(cut, condition=Condition.B)


def test_the_world_rules_carry_a_price_guide() -> None:
    """Pilot: with no anchor, real agents sold a bow at 1.2 gold."""
    from rpg_economy.agents.prompts import render_world

    config = make_config()
    text = render_world(config)
    assert "Price guide" in text and "not an official price" in text
    for good, price in config.economy.initial_reference_price.items():
        assert f"{good} {price}" in text


def test_the_artisan_brief_says_gold_comes_only_from_selling() -> None:
    """Pilot: real artisans crafted and kept everything, and went broke."""
    from rpg_economy.agents.prompts import ROLE_BRIEF
    from rpg_economy.domain import Role

    brief = ROLE_BRIEF[Role.ARTISAN]
    assert "only from selling what you craft" in brief
    assert "storage fee every tick" in brief


def test_an_artisans_world_rules_leave_out_gathering() -> None:
    """Zones, rest and wear are only noise to an artisan; the rest stays."""
    from rpg_economy.agents.prompts import render_world
    from rpg_economy.domain import Role

    config = make_config()
    artisan = render_world(config, Role.ARTISAN)
    adventurer = render_world(config, Role.ADVENTURER)
    director = render_world(config)
    for text in (artisan, adventurer, director):
        assert "Recipes" in text and "Price guide" in text and "Costs:" in text
    assert "Zones:" not in artisan and "Equipment wear" not in artisan
    assert "Zones:" in adventurer and "Equipment wear" in adventurer
    assert director == adventurer


def test_headlines_name_wanted_raw_materials_and_missing_inputs(rig) -> None:
    """2.29: adventurers sat on raw materials buyers bid for, and
    smiths without iron never bid for it."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig

    def state(agent: str) -> str:
        observation = build_observation(config, engine, agent, messages=messages)
        return build_agent_prompt(observation, config).sections["state"]

    engine.state.account("smithing_2").gold += Decimal("100")
    engine.submit_order("smithing_2", OrderSide.BUY, Resource.IRON, 2, "10.00")
    ranger = engine.state.account("ranger_1")
    ranger.inventory = {Resource.IRON: 2, Product.HEALTH_POTION: 1}
    ranger_state = state("ranger_1")
    assert (
        "Wanted: iron: you hold 2 and nobody else offers it; buyers bid 10.00 so far, and you "
        "set the price"
    ) in ranger_state
    observation = build_observation(config, engine, "ranger_1", messages=messages)
    market = build_agent_prompt(observation, config).sections["market"]
    assert "raw materials: you only sell them; the market refuses your bids" in market
    signals = build_observation(config, engine, "ranger_1", messages=messages).signals
    assert any(line.startswith("- iron (you only sell it): SHORT") for line in signals)

    engine.state.account("warrior_1").gold += Decimal("50")
    engine.submit_order("warrior_1", OrderSide.BUY, Product.MELEE_WEAPON, 1, "46.00")
    smith = engine.state.account("smithing_1")
    smith.inventory = {Resource.LEATHER: 1}
    guide = config.economy.initial_reference_price["iron"]
    assert (
        "Wanted: melee_weapon: buyers bid 46.00 and nobody sells it; each run needs 2 iron "
        f"(nobody offers any: bid about {guide:.2f}), which you lack, and your capacity allows"
    ) in state("smithing_1")
    # Told only "bid for the inputs", alchemists bid the guide below the ask.
    engine.state.account("ranger_2").receive(Resource.IRON, 2)
    engine.submit_order("ranger_2", OrderSide.SELL, Resource.IRON, 2, "10.40")
    assert "each run needs 2 iron (offered at 10.40: a bid at that price buys now)" in state(
        "smithing_1"
    )


def test_an_offer_at_or_below_cost_is_not_told_to_ask_less(rig) -> None:
    """2.29: told only "ask less", smiths cut armor worth 48 to 41."""
    from rpg_economy.domain import OrderSide

    config, engine, messages = rig
    engine.state.account("smithing_2").receive(Product.ARMOR, 1)
    engine.submit_order("smithing_2", OrderSide.SELL, Product.ARMOR, 1, "30.00")
    observation = build_observation(config, engine, "smithing_1", messages=messages)
    armor = next(line for line in observation.signals if line.startswith("- armor:"))
    assert "offered at 30.00 with no bids, at or below its cost" in armor
    assert "ask less" not in armor


def test_stock_already_on_its_own_orders_is_named(rig) -> None:
    """2.30: artisans re-offered goods already on their sell orders."""
    from rpg_economy.domain import OrderSide

    config, engine, messages = rig
    engine.state.account("smithing_1").receive(Product.ARMOR, 2)
    engine.submit_order("smithing_1", OrderSide.SELL, Product.ARMOR, 2, "52.00")
    observation = build_observation(config, engine, "smithing_1", messages=messages)
    armor = next(line for line in observation.signals if line.startswith("- armor:"))
    assert "all 2 you hold are on your own sell orders: cancel one to reprice" in armor


def test_an_expired_bid_says_bid_higher_only_when_nothing_is_offered(rig) -> None:
    """Told "bid higher" while sellers offered, buyers chased every ask
    and raw prices never came back once supply returned."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    alchemist = engine.state.account("alchemy_1")
    alchemist.gold += Decimal("100")
    engine.submit_order("alchemy_1", OrderSide.BUY, Resource.HERBS, 1, "5.00")
    for _ in range(config.market.order_expiry_ticks + 1):
        engine.run_cycle()

    def herbs_line() -> str:
        signals = build_observation(config, engine, "alchemy_1", messages=messages).signals
        return next(line for line in signals if line.startswith("- herbs:"))

    assert "expired with nothing bought: bid about 5.50 to get any" in herbs_line()

    engine.state.account("ranger_1").receive(Resource.HERBS, 1)
    engine.submit_order("ranger_1", OrderSide.SELL, Resource.HERBS, 1, "9.00")
    line = herbs_line()
    assert "expired with nothing bought; sellers ask 9.00" in line
    assert "bid higher" not in line


def test_a_buyer_whose_bids_filled_is_told_a_lower_bid_may_fill(rig) -> None:
    """Once supply returned, buyers kept paying the shock price and
    the herb price never came back."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    engine.state.account("alchemy_1").gold += Decimal("100")
    engine.state.account("ranger_1").receive(Resource.HERBS, 2)
    engine.submit_order("ranger_1", OrderSide.SELL, Resource.HERBS, 1, "10.00")
    engine.submit_order("alchemy_1", OrderSide.BUY, Resource.HERBS, 1, "10.00")

    def herbs_line(agent: str) -> str:
        signals = build_observation(config, engine, agent, messages=messages).signals
        return next((line for line in signals if line.startswith("- herbs")), "")

    assert "may also fill" not in herbs_line("alchemy_1")
    engine.submit_order("ranger_1", OrderSide.SELL, Resource.HERBS, 1, "10.00")
    assert (
        "your bids filled and sellers still offer at 10.00: a bid of about 8.00 may also fill"
    ) in herbs_line("alchemy_1")
    # Only artisans buying inputs: adventurers' gear purchases are never pushed down.
    assert "may also fill" not in herbs_line("ranger_2")


def test_wanted_is_not_shown_while_the_stock_limit_blocks_the_craft(rig) -> None:
    """The line "your inputs allow 1 run now" sat two lines above "not allowed"."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    alchemist = engine.state.account("alchemy_1")
    alchemist.receive(Product.HEALTH_POTION, 2)
    alchemist.receive(Resource.HERBS, 1)
    engine.state.account("mage_1").gold += Decimal("20")
    engine.submit_order("mage_1", OrderSide.BUY, Product.HEALTH_POTION, 1, "10.00")
    status = build_observation(config, engine, "alchemy_1", messages=messages).status
    assert not any(x.startswith("Wanted: health_potion") for x in status)
    assert any(x.startswith("Ready to sell: health_potion x2") for x in status)


def test_an_adventurer_sees_how_crowded_each_zone_is_and_what_a_trip_is_worth(rig) -> None:
    """22 trips of 23 went to the safe zone, never told what the others paid."""
    config, engine, messages = rig
    status = build_observation(config, engine, "warrior_1", messages=messages).status
    (line,) = [x for x in status if x.startswith("Zones now: ")]
    for zone in ("green_hollow", "iron_marches", "arcane_deeps"):
        assert f"{zone}: 0 trips set out in the last 30 ticks (even, success +0.00)" in line
    assert "green_hollow: 0 trips" in line and "a success likely finds about 3 units and" in line
    assert " gold" in line
    assert "a tick with the rest after it" in line
    # Artisans do not go on expeditions.
    artisan = build_observation(config, engine, "smithing_1", messages=messages).status
    assert not any(x.startswith("Zones now") for x in artisan)


def test_the_world_says_what_each_zone_yields_most_and_that_crowds_share_it(rig) -> None:
    from rpg_economy.agents.prompts import render_world
    from rpg_economy.domain import Role

    config, _engine, _messages = rig
    world = render_world(config, Role.ADVENTURER)
    assert "green_hollow: risk low, 2 ticks" in world and "mostly herbs (" in world
    assert "a success carries up to 7 units, as many as the zone holds" in world
    assert "Its gold is limited the same way" in world
    assert "trips that arrive together share it" in world
    assert "0.02 less success for each trip over that share, up to 0.15" in world


def test_a_narrow_raw_material_spread_suggests_meeting_halfway(rig) -> None:
    """Alchemists bid 6.00 and adventurers asked 6.60 for herbs, and no
    herb traded for 30 ticks."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    engine.state.account("ranger_1").receive(Resource.HERBS, 2)
    engine.submit_order("ranger_1", OrderSide.SELL, Resource.HERBS, 2, "6.60")
    engine.state.account("alchemy_2").gold += Decimal("20")
    engine.submit_order("alchemy_2", OrderSide.BUY, Resource.HERBS, 1, "6.00")

    def line(agent: str, good: str) -> str:
        signals = build_observation(config, engine, agent, messages=messages).signals
        return next(x for x in signals if x.startswith(f"- {good}"))

    assert "bid 6.00, offered 6.60: a deal at about 6.30, halfway, would trade" in line(
        "alchemy_1", "herbs"
    )
    # A wide spread, and products, get no midpoint.
    engine.state.account("ranger_2").receive(Resource.IRON, 1)
    engine.submit_order("ranger_2", OrderSide.SELL, Resource.IRON, 1, "20.00")
    engine.state.account("smithing_2").gold += Decimal("20")
    engine.submit_order("smithing_2", OrderSide.BUY, Resource.IRON, 1, "10.00")
    assert "halfway" not in line("smithing_1", "iron")
    engine.state.account("smithing_2").receive(Product.ARMOR, 1)
    engine.submit_order("smithing_2", OrderSide.SELL, Product.ARMOR, 1, "66.00")
    engine.state.account("mage_2").gold += Decimal("70")
    engine.submit_order("mage_2", OrderSide.BUY, Product.ARMOR, 1, "63.00")
    assert "halfway" not in line("warrior_1", "armor")


def test_a_bid_at_input_cost_is_not_a_reason_to_sell(rig) -> None:
    """Told a 6.00 bid was "above its 6.00 input cost", alchemists sold
    every potion for the price of its herb."""
    from rpg_economy.domain import OrderSide

    config, engine, messages = rig
    engine.state.account("alchemy_1").receive(Product.HEALTH_POTION, 1)
    mage = engine.state.account("mage_1")
    mage.gold += Decimal("50")
    engine.submit_order("mage_1", OrderSide.BUY, Product.HEALTH_POTION, 1, "6.00")

    def seen() -> tuple[list[str], str]:
        observation = build_observation(config, engine, "alchemy_1", messages=messages)
        line = next(x for x in observation.signals if x.startswith("- health_potion"))
        return observation.status, line

    status, line = seen()
    assert not any(x.startswith("Ready to sell") for x in status)
    assert not any("sell some" in x for x in status)
    assert "the only bid is 6.00, equal to what one costs you: selling there earns nothing" in line
    # Told holding costs storage but never how little, they sold "to reduce storage".
    assert "holding one costs 0.03 a tick, so waiting for a buyer near its 10.00 reference" in line

    engine.submit_order("mage_1", OrderSide.BUY, Product.HEALTH_POTION, 1, "7.00")
    status, line = seen()
    assert any(x.startswith("Ready to sell: health_potion x1: buyers bid 7.00") for x in status)


def test_a_price_far_from_its_typical_level_says_both(rig) -> None:
    """The last trades were all the shock's, and herbs stayed at 2.6x
    base through the recovery."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    assert config.market.usual_price_ticks == 340

    def trade(price: str) -> None:
        engine.state.account("ranger_1").receive(Resource.HERBS, 1)
        engine.state.account("alchemy_1").gold += Decimal("20")
        engine.submit_order("ranger_1", OrderSide.SELL, Resource.HERBS, 1, price)
        engine.submit_order("alchemy_1", OrderSide.BUY, Resource.HERBS, 1, price)

    def herbs_line() -> str:
        signals = build_observation(config, engine, "alchemy_2", messages=messages).signals
        return next((line for line in signals if line.startswith("- herbs")), "")

    # Too few trades for a typical price, so the guide stands in.
    trade("3.00")
    assert "the price guide puts it at 6.00; the latest trade was 3.00" in herbs_line()

    for price in ("6.30", "6.30", "6.30", "6.30", "6.30", "17.00"):
        trade(price)
    assert "it typically traded at 6.30 over the last 340 ticks; the latest trade was 17.00" in (
        herbs_line()
    )
    trade("7.00")
    line = herbs_line()
    assert "typically" not in line
    assert "price guide" not in line


def test_the_artisan_brief_says_a_sale_at_cost_earns_nothing(rig) -> None:
    """Told a held product "earns nothing and pays storage", alchemists
    sold every potion at its input cost to clear stock."""
    from rpg_economy.agents.prompts import ROLE_BRIEF
    from rpg_economy.domain import Role

    brief = ROLE_BRIEF[Role.ARTISAN]
    assert "a sale at or below that cost earns nothing" in brief
    assert "small storage fee" in brief and "so hold out for one" in brief


def test_the_rules_read_a_price_reference_as_a_reference_not_a_limit(rig) -> None:
    """Agents should know the reference without refusing to trade away from it."""
    from rpg_economy.agents.prompts import RULES

    assert "a reference for what it is worth, not a limit" in RULES
    assert "above the reference if that is what it takes to trade" in RULES
    assert "not from the guide" not in RULES


def test_a_price_signal_suggests_a_step_and_never_below_cost(rig) -> None:
    """Told only "ask less", sellers cut 0.05 at a time and the herb
    price took the whole recovery to fall 3%."""
    from rpg_economy.domain import OrderSide, Resource

    config, engine, messages = rig
    assert config.market.price_step_pct == Decimal("0.10")
    engine.state.account("ranger_2").receive(Resource.HERBS, 1)
    engine.submit_order("ranger_2", OrderSide.SELL, Resource.HERBS, 1, "8.00")
    ranger = build_observation(config, engine, "ranger_1", messages=messages).signals
    assert (
        "- herbs (you only sell it): offered at 8.00 with no bids: a seller who wants to "
        "sell must ask about 7.00"
    ) in ranger

    # An artisan's product: the step stops at the input cost.
    from rpg_economy.agents.observation import unit_cost

    smith = engine.state.account("smithing_2")
    smith.receive(Product.ARMOR, 1)
    cost = unit_cost(config, engine.market, Product.ARMOR)
    price = (cost * Decimal("1.05")).quantize(Decimal("0.01"))
    engine.submit_order("smithing_2", OrderSide.SELL, Product.ARMOR, 1, str(price))
    armor = next(
        line
        for line in build_observation(config, engine, "smithing_1", messages=messages).signals
        if line.startswith("- armor:")
    )
    assert f"must ask less, but not below its {cost} cost" in armor


def test_a_raw_material_far_from_its_typical_price_moves_back_halfway(rig) -> None:
    """10% at a time, herbs took 70 ticks to come from 14 to 10 against
    a 6.00 typical price with sellers offering."""
    from rpg_economy.agents.observation import moved
    from rpg_economy.domain import Resource

    config, engine, _ = rig
    # Dear and offered: halfway down, never under the typical price.
    assert moved(config, engine, Resource.HERBS, Decimal("14.00"), False) == Decimal("10.00")
    assert moved(config, engine, Resource.HERBS, Decimal("6.50"), False) == Decimal("6.00")
    # Cheap and short: halfway up.
    assert moved(config, engine, Resource.HERBS, Decimal("3.00"), True) == Decimal("4.50")
    # Away from it, or near it, one step.
    assert moved(config, engine, Resource.HERBS, Decimal("6.00"), True) == Decimal("6.60")
    assert moved(config, engine, Resource.HERBS, Decimal("4.00"), False) == Decimal("3.60")
    # Products follow their inputs, not this.
    assert moved(config, engine, Product.HEALTH_POTION, Decimal("20.00"), False) == Decimal("18.00")


def test_a_buyer_facing_a_dear_ask_is_told_the_last_sale_not_the_sellers_cut(rig) -> None:
    """Herbs sold at 5.40, one seller asked 11.39, and an alchemist told
    "a seller who wants to sell must ask about 8.70" bid 8.70."""
    from rpg_economy.agents.observation import market_signals
    from rpg_economy.domain import OrderSide, Resource

    config, engine, _ = rig
    engine.state.account("warrior_1").receive(Resource.HERBS, 3)
    engine.state.account("alchemy_2").gold += Decimal("100")
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.HERBS, 1, "5.40")
    engine.submit_order("alchemy_2", OrderSide.BUY, Resource.HERBS, 1, "5.40")
    engine.submit_order("warrior_1", OrderSide.SELL, Resource.HERBS, 2, "11.39")
    alchemist = engine.state.account("alchemy_1")
    (line,) = [
        x for x in market_signals(config, engine, alchemist, [Resource.HERBS]) if "herbs" in x
    ]
    assert "offered at 11.39 with no bids, above its last sale at 5.40" in line
    assert "a bid about 5.40 may fill" in line
    assert "must ask" not in line
    # The seller is still told to cut.
    ranger = engine.state.account("ranger_1")
    (line,) = [x for x in market_signals(config, engine, ranger, [Resource.HERBS]) if "herbs" in x]
    assert "a seller who wants to sell must ask about 8.70" in line
