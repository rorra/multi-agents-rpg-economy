"""Configuration loading, layering, validation and hashing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rpg_economy.config import ConfigError, config_hash, config_payload, deep_merge, load_config
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
