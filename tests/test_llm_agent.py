"""LLM-backed activations: prompts, parsing, retry policy, token buckets."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.agents.llm_agent import LLMAgentRunner
from rpg_economy.agents.observation import build_observation
from rpg_economy.agents.prompts import (
    PROMPT_VERSION,
    build_agent_prompt,
    prompt_hashes,
)
from rpg_economy.domain import Product
from rpg_economy.engine import Engine, build_world
from rpg_economy.llm import MockLLMClient, TokenLedger
from rpg_economy.logging_ import EventLogWriter
from rpg_economy.messaging import MessageBus
from rpg_economy.orchestration import build_directory
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
