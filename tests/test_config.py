"""Configuration loading, layering, validation and hashing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rpg_economy.config import (
    ConfigError,
    config_hash,
    config_payload,
    deep_merge,
    load_config,
)
from tests.conftest import BASE_CONFIG

BASE = BASE_CONFIG


def test_base_config_loads() -> None:
    config = load_config(BASE)
    population = config.population
    # The calibrated population: two of each subclass and specialization.
    assert {
        k.value if hasattr(k, "value") else k: v for k, v in population.adventurers.items()
    } == {"warrior": 2, "mage": 2, "ranger": 2}
    assert {k.value if hasattr(k, "value") else k: v for k, v in population.artisans.items()} == {
        "smithing": 2,
        "woodworking": 2,
        "alchemy": 2,
    }
    assert population.total_agents == 12
    assert config.time.k_activations >= 1


def test_phase_ticks_must_sum_to_horizon() -> None:
    with pytest.raises(ConfigError, match="horizon_ticks"):
        load_config(BASE, overrides={"phases": {"base_ticks": 1}})


def test_unknown_key_is_rejected() -> None:
    """A silently ignored typo would run the experiment under unintended parameters."""
    with pytest.raises(ConfigError):
        load_config(BASE, overrides={"time": {"k_activation": 5}})


def test_pilot_and_final_seeds_must_be_disjoint() -> None:
    with pytest.raises(ConfigError, match="disjoint"):
        load_config(BASE, overrides={"seeds": {"pilot": [1], "final": [1, 2]}})


@pytest.mark.parametrize("which", ["pilot", "final"])
def test_a_seed_list_must_not_repeat_a_seed(which: str) -> None:
    seeds = {"pilot": [1, 2], "final": [3, 4]}
    seeds[which] = [*seeds[which], seeds[which][0]]
    with pytest.raises(ConfigError, match=f"seeds.{which} has duplicates"):
        load_config(BASE, overrides={"seeds": seeds})


def test_the_primary_scenario_is_not_a_secondary_one() -> None:
    with pytest.raises(ConfigError, match="primary scenario"):
        load_config(
            BASE,
            overrides={"campaign": {"secondary_scenarios": ["demand_surge", "resource_scarcity"]}},
        )


def test_secondary_scenarios_must_not_repeat() -> None:
    with pytest.raises(ConfigError, match="duplicates: demand_surge"):
        load_config(
            BASE,
            overrides={"campaign": {"secondary_scenarios": ["demand_surge", "demand_surge"]}},
        )


def test_the_schema_primary_scenario_matches_the_campaign_plan() -> None:
    """models.py cannot import the plan (the plan imports it), so the two are pinned here."""
    from rpg_economy.campaign.plan import PRIMARY_SCENARIO
    from rpg_economy.config import models

    assert models._PRIMARY_SCENARIO is PRIMARY_SCENARIO


def test_liquidity_loss_names_only_agents_the_population_has() -> None:
    with pytest.raises(ConfigError, match=r"does not have: \['smithing_9'\]"):
        load_config(
            BASE,
            overrides={"shocks": {"liquidity_loss": {"affected_agents": ["smithing_9"]}}},
        )


def test_liquidity_loss_must_not_name_an_agent_twice() -> None:
    with pytest.raises(ConfigError, match="duplicates"):
        load_config(
            BASE,
            overrides={
                "shocks": {"liquidity_loss": {"affected_agents": ["warrior_1", "warrior_1"]}}
            },
        )


def test_experiment_2_requires_a_budget() -> None:
    with pytest.raises(ConfigError, match="experiment_2_total_tokens"):
        load_config(BASE, overrides={"run": {"experiment": "experiment_2"}})


def test_experiment_2_accepts_a_derived_budget() -> None:
    config = load_config(
        BASE,
        overrides={
            "run": {"experiment": "experiment_2"},
            "budget": {"experiment_2_total_tokens": 500_000},
        },
    )
    assert config.budget.experiment_2_total_tokens == 500_000


def test_a_budget_records_whether_it_was_derived_from_the_mock() -> None:
    assert load_config(BASE).budget.derived_from_mock is False
    budget = {"experiment_2_total_tokens": 500_000}
    real = load_config(BASE, overrides={"budget": budget})
    mock = load_config(BASE, overrides={"budget": {**budget, "derived_from_mock": True}})
    assert mock.budget.derived_from_mock is True
    # A mock-derived budget is a different setup from a real-derived one.
    assert config_hash(mock) != config_hash(real)


def test_recovery_reference_window_must_fit_in_the_base_phase() -> None:
    """The reference value is the mean over the last W base-phase ticks (design s11.1)."""
    with pytest.raises(ConfigError, match="volatility_window_w"):
        load_config(BASE, overrides={"metrics": {"volatility_window_w": 9999}})


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def test_hash_is_stable_across_reloads() -> None:
    assert config_hash(load_config(BASE)) == config_hash(load_config(BASE))


def test_hash_is_stable_across_processes() -> None:
    """Set and dict ordering vary with PYTHONHASHSEED; the hash must not."""
    import os
    import subprocess
    import sys

    code = (
        "from rpg_economy.config import load_config, config_hash;"
        f"print(config_hash(load_config({str(BASE)!r})))"
    )
    digests = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        env.pop("OPENAI_API_KEY", None)
        done = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(done.stdout.strip())
    assert digests == {config_hash(load_config(BASE))}


def test_hash_ignores_yaml_key_order(tmp_path: Path) -> None:
    """A cosmetically reordered config must not look like a different experiment."""
    payload = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    reordered = dict(reversed(list(payload.items())))
    shuffled = tmp_path / "shuffled.yaml"
    shuffled.write_text(yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8")

    assert config_hash(load_config(shuffled)) == config_hash(load_config(BASE))


def test_hash_changes_when_a_parameter_changes() -> None:
    baseline = config_hash(load_config(BASE))
    altered = config_hash(load_config(BASE, overrides={"time": {"k_activations": 12}}))
    assert altered != baseline


def test_hash_is_insensitive_to_comments_and_indentation(tmp_path: Path) -> None:
    text = BASE.read_text(encoding="utf-8")
    stripped = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    path = tmp_path / "nocomments.yaml"
    path.write_text(stripped, encoding="utf-8")
    assert config_hash(load_config(path)) == config_hash(load_config(BASE))


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
def test_deep_merge_is_recursive_and_non_mutating() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": [1, 2]}
    snapshot = json.dumps(base, sort_keys=True)
    merged = deep_merge(base, {"a": {"c": 3}, "d": [9]})

    assert merged == {"a": {"b": 1, "c": 3}, "d": [9]}
    assert json.dumps(base, sort_keys=True) == snapshot, "deep_merge mutated its input"


def test_layers_apply_in_order(tmp_path: Path) -> None:
    first = tmp_path / "one.yaml"
    second = tmp_path / "two.yaml"
    first.write_text("time: {k_activations: 5}", encoding="utf-8")
    second.write_text("time: {k_activations: 7}", encoding="utf-8")

    config = load_config(BASE, layers=[first, second])
    assert config.time.k_activations == 7


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


def test_config_payload_is_json_serializable() -> None:
    json.dumps(config_payload(load_config(BASE)))


def test_initial_inventory_enters_the_config_hash() -> None:
    """It is part of the endowment design s12 holds constant across a trio, so a
    change to it must be as visible in the run record as a change to gold."""
    baseline = config_hash(load_config(BASE))
    altered = config_hash(
        load_config(BASE, overrides={"economy": {"initial_inventory": {"smithing": {"iron": 99}}}})
    )
    assert altered != baseline


def test_every_endowment_parameter_is_hashed() -> None:
    payload = config_payload(load_config(BASE))["economy"]
    for parameter in ("initial_gold", "initial_inventory", "initial_reference_price"):
        assert parameter in payload


def test_the_execution_mode_is_not_part_of_the_config_hash() -> None:
    """Mock or real is how a setup runs, not what it is: one identity for both."""
    from rpg_economy.cli import with_real_provider

    config = load_config(BASE)
    real = with_real_provider(config)

    assert config.llm.mock is True
    assert real.llm.mock is False
    assert "mock" not in config_payload(config)["llm"]
    assert config_payload(real) == config_payload(config)
    assert config_hash(real) == config_hash(config)


def test_switching_to_the_real_provider_changes_nothing_else() -> None:
    from rpg_economy.cli import with_real_provider

    config = load_config(BASE)
    real = with_real_provider(config)

    assert real is not config
    assert config.llm.mock is True, "the original config must not be mutated"
    assert real.model_dump(exclude={"llm": {"mock"}}) == config.model_dump(
        exclude={"llm": {"mock"}}
    )


def test_the_population_has_only_adventurers_and_artisans() -> None:
    """There is no merchant role: a configuration that asks for one is refused."""
    from rpg_economy.domain import Role

    assert {role.value for role in Role} == {"adventurer", "artisan"}
    with pytest.raises(ConfigError):
        load_config(BASE, overrides={"population": {"merchants": 2}})


def test_a_recipe_input_must_be_at_least_one_unit() -> None:
    """A zero input would divide by zero when the rule-based artisan sizes a batch."""
    with pytest.raises(ConfigError, match="recipes"):
        load_config(
            BASE, overrides={"economy": {"recipes": {"melee_weapon": {"inputs": {"iron": 0}}}}}
        )


def test_durability_must_be_at_least_one_point() -> None:
    with pytest.raises(ConfigError, match="durability"):
        load_config(BASE, overrides={"economy": {"durability": {"armor": 0}}})


def test_the_demand_surge_target_must_not_be_equipment() -> None:
    """Durable gear is not used up per trip, so a surge on it would change nothing."""
    with pytest.raises(ConfigError, match="durable equipment"):
        load_config(
            BASE, overrides={"shocks": {"demand_surge": {"target_product": "melee_weapon"}}}
        )


def test_the_demand_surge_target_must_be_carried_by_some_subclass() -> None:
    from pydantic import ValidationError

    from rpg_economy.config.models import SimulationConfig
    from rpg_economy.domain import Product

    payload = load_config(BASE).model_dump(mode="python")
    carriers = list(payload["expeditions"]["gear"].values())
    assert any(Product.MANA_POTION in items for items in carriers)
    for items in carriers:
        items.pop(Product.MANA_POTION, None)
    payload["shocks"]["demand_surge"]["target_product"] = Product.MANA_POTION
    with pytest.raises(ValidationError, match="in no subclass's expedition gear"):
        SimulationConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Run phase
# ---------------------------------------------------------------------------
def test_the_run_phase_defaults_to_adhoc() -> None:
    assert load_config(BASE).run.phase == "adhoc"


@pytest.mark.parametrize("phase", ["pilot", "final", "adhoc"])
def test_for_cell_sets_the_run_phase(phase: str) -> None:
    from rpg_economy.domain import ExperimentRegime, Scenario

    config = load_config(BASE)
    cell = config.for_cell(
        scenario=Scenario.DEMAND_SURGE,
        experiment=ExperimentRegime.EQUAL_ACTIVITY,
        seed=7,
        phase=phase,
    )
    assert cell.run.phase == phase
    assert cell.run.seed == 7
    # Without a phase, for_cell keeps the one the config has.
    kept = cell.for_cell(
        scenario=Scenario.RESOURCE_SCARCITY, experiment=ExperimentRegime.EQUAL_ACTIVITY, seed=8
    )
    assert kept.run.phase == phase


def test_an_unknown_run_phase_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(BASE, overrides={"run": {"phase": "calibration"}})


# ---------------------------------------------------------------------------
# Load-time validation of what would otherwise fail mid-run
# ---------------------------------------------------------------------------
def test_a_recipe_must_have_at_least_one_input() -> None:
    from rpg_economy.config.models import SimulationConfig

    payload = load_config(BASE).model_dump(mode="json")
    # An override would merge into the base inputs, so empty them in place.
    next(iter(payload["economy"]["recipes"].values()))["inputs"] = {}
    with pytest.raises(ValueError, match="inputs"):
        SimulationConfig.model_validate(payload)


def test_the_scarce_resource_must_spawn_somewhere() -> None:
    config = load_config(BASE)
    target = config.shocks.resource_scarcity.target_resource
    zones = [
        {
            **zone.model_dump(mode="json"),
            "spawn_per_tick": {
                r.value: str(rate) for r, rate in zone.spawn_per_tick.items() if r is not target
            },
        }
        for zone in config.expeditions.zones
    ]
    with pytest.raises(ConfigError, match="spawns in no zone"):
        load_config(BASE, overrides={"expeditions": {"zones": zones}})


def test_the_productivity_drop_target_must_have_an_artisan() -> None:
    config = load_config(BASE)
    target = config.shocks.productivity_drop.target_specialization.value
    with pytest.raises(ConfigError, match="has no artisan"):
        load_config(BASE, overrides={"population": {"artisans": {target: 0}}})


def test_the_productivity_drop_target_must_have_capacity() -> None:
    config = load_config(BASE)
    target = config.shocks.productivity_drop.target_specialization.value
    with pytest.raises(ConfigError, match="no production capacity"):
        load_config(
            BASE,
            overrides={"economy": {"production_capacity_per_cycle": {target: 0}}},
        )


@pytest.mark.parametrize(
    "table",
    [
        "duration_ticks",
        "recovery_ticks_on_failure",
        "success_probability",
        "expected_yield",
        "yield_spread",
    ],
)
def test_every_zone_risk_needs_an_entry_in_each_risk_table(table: str) -> None:
    config = load_config(BASE)
    risk = config.expeditions.zones[0].risk
    entries = {
        r.value: str(v) for r, v in getattr(config.expeditions, table).items() if r is not risk
    }
    # The override merges into the base mapping, so replace the whole section.
    expeditions = config.expeditions.model_dump(mode="json")
    expeditions[table] = entries
    payload = config.model_dump(mode="json")
    payload["expeditions"] = expeditions
    from rpg_economy.config.models import SimulationConfig

    with pytest.raises(ValueError, match=f"expeditions.{table} has no entry"):
        SimulationConfig.model_validate(payload)


def test_an_interest_list_naming_an_unknown_good_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown good 'mithril'"):
        load_config(
            BASE, overrides={"observability": {"interest_lists": {"warrior": ["iron", "mithril"]}}}
        )


def test_the_budget_provenance_is_not_part_of_a_runs_identity() -> None:
    config = load_config(BASE)
    payload = config.model_dump(mode="python")
    payload["budget"] = {**payload["budget"], "derived_code_digest": "abc123"}
    stamped = type(config).model_validate(payload)
    assert config_hash(stamped) == config_hash(config)
