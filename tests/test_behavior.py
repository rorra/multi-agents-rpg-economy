"""The behaviour check: obligations, situations, scoring, audit and replay, offline."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from rpg_economy.behavior import (
    OBLIGATIONS,
    SITUATIONS,
    audit_log,
    build_jobs,
    estimate_behavior_cost,
    run_behavior_check,
)
from rpg_economy.behavior.obligations import BY_NAME, judge
from rpg_economy.behavior.replay import ReplayError, pick_moments, replay_jobs
from rpg_economy.behavior.runner import MAX_INVALID_SHARE, VALID_DECISION
from rpg_economy.domain import Condition
from rpg_economy.llm import MockLLMClient
from rpg_economy.runner import run_condition
from tests.conftest import make_config


@pytest.fixture(scope="module")
def jobs():
    return build_jobs(make_config())


def _job(jobs, name, condition=Condition.B):
    return next(j for j in jobs if j.name == name and j.condition is condition)


def _sell(good, price, quantity=1):
    return {"action": "sell", "target": good, "quantity": quantity, "price": str(price)}


def _buy(good, price, quantity=1):
    return {"action": "buy", "target": good, "quantity": quantity, "price": str(price)}


def _orders(*orders):
    return {"action": "place_orders", "orders": list(orders)}


WAIT = {"action": "wait"}


def test_every_situation_builds_and_its_focus_applies(jobs) -> None:
    assert len(jobs) == sum(len(s.conditions) for s in SITUATIONS)
    assert len({s.name for s in SITUATIONS}) == len(SITUATIONS)
    for job in jobs:
        assert job.system and job.user
        missing = [f for f in job.situation.focus if f not in job.obligations()]
        assert not missing, (job.name, job.condition, missing)


def test_every_role_rule_is_the_focus_of_some_situation() -> None:
    focused = {f for s in SITUATIONS for f in s.focus}
    general = {"adv_buys_only_its_gear", "adv_equips_only_held"}
    general |= {"art_produces_only_possible", "buys_within_gold", "no_self_cross"}
    assert {o.name for o in OBLIGATIONS} - general <= focused


def test_variants_carry_the_noise_and_c_the_bulletin(jobs) -> None:
    clean = _job(jobs, "adv_resting_holds_wanted_raw:clean").user
    noisy = _job(jobs, "adv_resting_holds_wanted_raw:noisy").user
    assert "health_potion" in noisy and '"your_open_orders":[]' in clean
    assert "Global bulletin" in _job(jobs, "adv_resting_holds_wanted_raw:clean", Condition.C).user
    assert "Global bulletin" not in clean
    assert "ANALYZE_MARKET" in _job(jobs, "task_analyze_market_herbs", Condition.A).user


def test_the_adventurer_rules_tell_wanted_from_unwanted_answers(jobs) -> None:
    view = _job(jobs, "adv_resting_holds_wanted_raw:noisy").view
    assert judge(view, _sell("iron", 10))["adv_sells_wanted_raw"]
    assert judge(view, _orders(_sell("iron", 12), _sell("wood", 9)))["adv_sells_wanted_raw"]
    assert not judge(view, _sell("health_potion", 4))["adv_sells_wanted_raw"]
    assert not judge(view, {"action": "start_expedition", "target": "green_hollow"})[
        "adv_no_set_out_when_unable"
    ]
    assert not judge(view, _buy("wood", 8))["adv_no_raw_buy"]
    # The pricing rule counts only answers that offer a scarce raw material.
    assert judge(view, _sell("iron", 12))["adv_asks_above_bid_when_scarce"]
    assert judge(view, _sell("iron", 10))["adv_asks_above_bid_when_scarce"] is False
    assert judge(view, _sell("iron", 30))["adv_asks_above_bid_when_scarce"] is False
    assert judge(view, WAIT)["adv_asks_above_bid_when_scarce"] is None

    gear = _job(jobs, "adv_broken_weapon_offered:resting").view
    assert judge(gear, _buy("melee_weapon", 39))["adv_buys_paying_gear"]
    assert not judge(gear, _buy("melee_weapon", 30))["adv_buys_paying_gear"]  # below the ask
    assert not judge(gear, WAIT)["adv_buys_paying_gear"]
    assert not judge(gear, {"action": "equip", "target": "melee_weapon"})["adv_equips_only_held"]
    assert not judge(gear, _buy("staff_focus", 60))["adv_buys_only_its_gear"]
    # Gear is optional: ready to go, setting out without it is as fair as buying.
    ready = _job(jobs, "adv_broken_weapon_offered:ready").view
    setting_out = judge(ready, {"action": "start_expedition", "target": "green_hollow"})
    assert setting_out["adv_ready_acts"] and "adv_buys_paying_gear" not in setting_out
    assert judge(ready, _buy("melee_weapon", 39))["adv_ready_acts"]

    poor = _job(jobs, "adv_broken_weapon_offered_holds_raw:poor").view
    assert "adv_buys_paying_gear" not in judge(poor, WAIT)  # it cannot pay the ask
    assert not judge(poor, _buy("melee_weapon", 39))["buys_within_gold"]

    unoffered = _job(jobs, "adv_broken_weapon_unoffered:resting").view
    assert judge(unoffered, _buy("melee_weapon", 40))["adv_bids_for_paying_gear"]
    assert not judge(unoffered, WAIT)["adv_bids_for_paying_gear"]

    ready = _job(jobs, "adv_ready_holds_wanted_raw:noisy").view
    assert judge(ready, {"action": "start_expedition", "target": "green_hollow"})["adv_ready_acts"]
    assert judge(ready, _sell("leather", 9.5))["adv_ready_acts"]
    assert not judge(ready, _sell("health_potion", 4))["adv_ready_acts"]


def test_the_artisan_rules_tell_wanted_from_unwanted_answers(jobs) -> None:
    weapons = _job(jobs, "art_weapons_wanted:clean").view
    produce = {"action": "produce", "quantity": 1}
    assert judge(weapons, {**produce, "target": "melee_weapon"})["art_crafts_wanted"]
    assert not judge(weapons, {**produce, "target": "armor"})["art_crafts_wanted"]

    no_iron = _job(jobs, "art_weapon_wanted_no_iron:noisy").view
    assert judge(no_iron, _buy("iron", 11, 2))["art_buys_input_for_wanted"]
    assert not judge(no_iron, _sell("armor", 41))["art_buys_input_for_wanted"]
    assert not judge(no_iron, {**produce, "target": "melee_weapon"})["art_produces_only_possible"]

    holds = _job(jobs, "art_holds_wanted_weapon:clean").view
    assert judge(holds, _sell("melee_weapon", 46))["art_sells_wanted_stock"]
    assert not judge(holds, _sell("melee_weapon", 90))["art_sells_wanted_stock"]  # not to the bid
    assert not judge(holds, WAIT)["art_sells_wanted_stock"]

    staff = _job(jobs, "art_staff_wanted_bow_lowball:clean").view
    assert judge(staff, _orders(_sell("staff_focus", 76), _sell("bow_gear", 40)))[
        "art_sells_wanted_stock"
    ]
    assert not judge(staff, _sell("bow_gear", 15.5))["art_no_sale_below_cost"]

    offered = _job(jobs, "art_stock_already_offered:clean").view
    assert not judge(offered, _sell("armor", 47))["sells_within_stock"]
    cancel_and_reprice = _orders({"action": "cancel_order", "order_id": 1}, _sell("armor", 47))
    assert judge(offered, cancel_and_reprice)["sells_within_stock"]
    assert not judge(offered, _buy("armor", 55))["no_self_cross"]

    glut = _job(jobs, "art_potion_glut:clean").view
    assert not judge(glut, {**produce, "target": "health_potion"})["art_stops_unsold"]
    assert judge(glut, _sell("health_potion", 4))["art_stops_unsold"]

    herbs = _job(jobs, "art_herb_bids_unfilled:clean").view
    assert judge(herbs, _buy("herbs", 6.5, 2))["art_raises_failed_bid"]
    assert not judge(herbs, _buy("herbs", 6, 2))["art_raises_failed_bid"]

    task = _job(jobs, "task_analyze_market_herbs", Condition.A).view
    assert not judge(task, WAIT)["task_answered_with_action"]


def test_rules_that_would_ask_two_things_at_once_give_way(jobs) -> None:
    """Holding a wanted product, an artisan is asked to sell it, not to craft more."""
    staff = _job(jobs, "art_staff_wanted_bow_lowball:clean").view
    names = set(judge(staff, WAIT))
    assert "art_sells_wanted_stock" in names
    assert "art_crafts_wanted" not in names and "art_buys_input_for_wanted" not in names


SELL_STAFF = (
    '{"action":"sell","target":"staff_focus","quantity":1,"price":76,'
    '"priority":"high","short_reason":"x"}'
)
REPLY_WAIT = '{"action":"wait","priority":"low","short_reason":"x"}'
DIRECTOR_CLOSE = '{"decision":"close"}'


def test_answers_are_judged_and_invalid_ones_counted(jobs) -> None:
    staff = [j for j in jobs if j.name == "art_staff_wanted_bow_lowball:clean"]
    director = [j for j in jobs if j.situation.director]
    replies = iter([SELL_STAFF, SELL_STAFF, REPLY_WAIT, "not json"] * 2 + [DIRECTOR_CLOSE] * 4)
    client = MockLLMClient(lambda system, user: next(replies))
    report = run_behavior_check(staff + director, client, reps=4, workers=1)

    results = [r for r in report.results if r.case == "art_staff_wanted_bow_lowball:clean"]
    for result in results:
        assert result.met("art_sells_wanted_stock") == 2 and result.invalid == 1
        assert "art_sells_wanted_stock" in result.failing  # 50% under 85%
        assert result.met("sells_within_stock") == 3  # the invalid answer meets nothing
    director_result = next(r for r in report.results if r.case == "director_answers_validly")
    assert director_result.passed and director_result.met(VALID_DECISION) == 4
    assert report.invalid_share == pytest.approx(2 / 12)
    assert report.invalid_share > MAX_INVALID_SHARE and not report.passed
    payload = report.to_payload()
    assert payload["calls"] == 12 and payload["passed"] is False
    assert payload["by_obligation"]["art_sells_wanted_stock"]["failed"] == 2


def test_the_check_is_priced_before_it_runs(jobs) -> None:
    pricing = make_config().llm.pricing
    one = estimate_behavior_cost(jobs, 1, pricing)
    eight = estimate_behavior_cost(jobs, 8, pricing)
    assert eight["calls"] == 8 * len(jobs)
    assert eight["input_tokens"] == 8 * one["input_tokens"]
    assert eight["total_cost"] >= one["total_cost"] > 0


SELL_HERBS = (
    '{"action":"sell","target":"herbs","quantity":1,"price":7,"priority":"low","short_reason":"x"}'
)


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory) -> Path:
    """A short B run whose answers vary, logged like a real one."""
    path = tmp_path_factory.mktemp("run") / "B.jsonl"
    counter = itertools.count()

    def answer(system: str, user: str) -> str:
        return SELL_HERBS if next(counter) % 3 == 0 else REPLY_WAIT

    config = make_config().for_condition(Condition.B, seed=7)
    run_condition(config, path, client=MockLLMClient(answer), cycles=12)
    return path


def _llm_calls(path: Path) -> list[dict]:
    events = (json.loads(line) for line in path.open(encoding="utf-8"))
    return [e for e in events if e["type"] == "llm_call"]


def test_the_audit_judges_every_activation_of_a_run(mock_run) -> None:
    report = audit_log(mock_run, make_config())
    assert report.condition == "B" and report.activations > 0
    assert report.tallies["sells_within_stock"].applicable == report.activations
    for tally in report.tallies.values():
        assert tally.met + len(tally.misses) == tally.applicable
    assert report.to_payload()["obligations"]["sells_within_stock"]["applicable"] > 0


def test_a_replay_rebuilds_the_logged_prompt_exactly(mock_run) -> None:
    call = _llm_calls(mock_run)[5]
    moment = (call["agent_id"], call["tick"])
    [job] = replay_jobs(mock_run, make_config(), [moment], label="mock")
    assert job.user == call["payload"]["user"]
    assert job.name == f"log:mock:{moment[0]}@{moment[1]}"
    assert job.view is not None and job.condition is Condition.B
    # Under another configuration the run cannot be rebuilt, unless that
    # configuration's past self is offered (configs/history).
    other = make_config(market={"order_expiry_ticks": 9})
    with pytest.raises(ReplayError):
        replay_jobs(mock_run, other, [moment], label="x")
    [job] = replay_jobs(mock_run, other, [moment], label="x", earlier=(make_config(),))
    assert job.user == call["payload"]["user"]


def test_moments_are_picked_from_the_misses(mock_run) -> None:
    report = audit_log(mock_run, make_config())
    picked = pick_moments(report, 2)
    misses = {m for t in report.tallies.values() for m in t.misses}
    assert set(picked) <= misses
    assert len(picked) == len(set(picked))
    assert pick_moments(report, 0) == []


def test_the_command_prices_and_lists_without_asking(capsys) -> None:
    from rpg_economy.cli import main

    assert main(["behavior-check", "--only", "adv_resting_holds_wanted_raw"]) == 0
    out = capsys.readouterr().out
    assert "adv_resting_holds_wanted_raw:noisy" in out and "adv_sells_wanted_raw" in out
    assert "art_potion_glut" not in out
    assert main(["behavior-check", "--only", "no_such_case"]) == 2
    # A real check is refused without the confirmed total.
    assert main(["behavior-check", "--real", "--spend-limit-set"]) == 5


def test_the_audit_command_reads_logs_only(capsys, tmp_path, mock_run) -> None:
    from rpg_economy.cli import main

    out = tmp_path / "audit.json"
    assert main(["behavior-audit", str(mock_run), "--out", str(out)]) == 0
    assert "sells_within_stock" in capsys.readouterr().out
    assert json.loads(out.read_text())[0]["condition"] == "B"


def test_every_obligation_has_a_threshold_and_a_role() -> None:
    for obligation in OBLIGATIONS:
        assert 0 < obligation.threshold <= 1
        assert obligation.role in {"adventurer", "artisan", "any"}
        assert BY_NAME[obligation.name] is obligation
