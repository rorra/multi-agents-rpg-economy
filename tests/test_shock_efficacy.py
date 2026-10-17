"""The shock-efficacy diagnostic: does each shock move its target? (design s10)"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.domain import ExperimentRegime, Scenario
from tests.conftest import make_config
from tests.test_report_pipeline import (  # noqa: F401 -- recorded_phase is an autouse fixture
    record_phase,
    recorded_phase,
    with_phase,
)


def _pilot_trio(
    config,
    directory: Path,
    seed: int,
    *,
    experiment: ExperimentRegime = ExperimentRegime.EQUAL_ACTIVITY,
    phase: str = "pilot",
):
    """One trio of the primary scenario, run as the pilot runs it."""
    from rpg_economy.runner import run_trio

    cell = with_phase(config, phase).for_cell(
        scenario=Scenario.RESOURCE_SCARCITY, experiment=experiment, seed=seed
    )
    record_phase(directory, phase)
    return run_trio(cell, directory, seed=seed, cycles=config.time.total_cycles)


SHORT = {
    "phases": {"base_ticks": 20, "shock_ticks": 10, "recovery_ticks": 10},
    "time": {"horizon_ticks": 40},
}


#: Rule-based adventurers buy every item for every trip: about 1.3x what LLM
#: adventurers used up, which the spawn rates are set from. At the
#: LLM rates the rule-based base phase already lacked herbs (88 of 90 control
#: ticks with none offered), leaving the shock nothing to add.
RULE_BASED_SUPPLY = Decimal("1.3")


def _rule_based_world(**overrides: object):
    base = make_config()
    zones = [
        {
            "id": zone.id,
            "risk": zone.risk.value,
            "spawn_per_tick": {
                r.value: str(rate * RULE_BASED_SUPPLY) for r, rate in zone.spawn_per_tick.items()
            },
        }
        for zone in base.expeditions.zones
    ]
    return make_config(expeditions={"zones": zones}, **overrides)


def test_the_scarcity_shock_cuts_the_supply_of_its_resource(tmp_path: Path) -> None:
    """The primary shock bites (design s10), in a world supplied for rule-based demand."""
    from rpg_economy.analysis.shock_efficacy import measure

    config = _rule_based_world(run={"scenario": Scenario.RESOURCE_SCARCITY.value})
    efficacy = measure(config, [1001, 1002, 1003], tmp_path)
    # The scarcity is deliberately mild, so stockouts keep headroom below the
    # whole shock phase; efficacy itself is judged on price and stockouts.
    assert efficacy.median("supply_change") <= -0.2
    assert efficacy.effective()


def test_an_ineffective_shock_is_reported_as_such() -> None:
    from rpg_economy.analysis.shock_efficacy import Efficacy, SeedEffect, ShockPhase

    same = ShockPhase(mean_price=10.0, stockout_ticks=5, traded_value=100.0, gathered=20)
    efficacy = Efficacy(scenario="resource_scarcity", target="iron")
    efficacy.effects = [SeedEffect(seed=s, shocked=same, control=same) for s in (1, 2, 3)]
    assert not efficacy.effective()


def test_pilot_logs_are_read_base_phase_against_shock_phase(tmp_path: Path) -> None:
    """No unshocked twin in a pilot: the run's own base phase is the reference."""
    from rpg_economy.analysis.shock_efficacy import from_logs
    from rpg_economy.logging_ import run_logs

    config = make_config(
        phases={"base_ticks": 20, "shock_ticks": 10, "recovery_ticks": 10},
        time={"horizon_ticks": 40},
    )
    _pilot_trio(config, tmp_path / "1001", 1001)

    groups, unmatched = from_logs(run_logs(tmp_path), config, allow_mock=True)
    assert unmatched == []
    assert set(groups) == {("resource_scarcity", c) for c in ("A", "B", "C")}
    for efficacy in groups.values():
        assert efficacy.target == "herbs"
        assert len(efficacy.effects) == 1
        effect = efficacy.effects[0]
        assert effect.shocked.gathered is not None and effect.control.gathered is not None


def test_the_efficacy_command_reads_pilot_logs(tmp_path: Path, capsys) -> None:
    from rpg_economy.cli import main

    assert main(["economy-check", "--shock-efficacy", "--pilot-logs", str(tmp_path)]) == 2
    assert "No finished runs" in capsys.readouterr().err


def test_the_pilot_control_is_the_last_w_base_ticks(tmp_path: Path) -> None:
    """Start-up ticks are the economy settling in, not the shock's control."""
    from rpg_economy.analysis.shock_efficacy import control_window, from_logs, window_stats
    from rpg_economy.logging_ import run_logs
    from rpg_economy.metrics.run_metrics import RunLog

    config = make_config(
        phases={"base_ticks": 20, "shock_ticks": 10, "recovery_ticks": 10},
        time={"horizon_ticks": 40},
        metrics={"volatility_window_w": 5},
    )
    assert control_window(config) == (15, 20)

    runs = _pilot_trio(config, tmp_path / "1001", 1001)
    groups, _ = from_logs(run_logs(tmp_path), config, allow_mock=True)

    log = RunLog(runs[next(iter(runs))].log_path)
    condition = log.metadata.condition.value
    control = groups[("resource_scarcity", condition)].effects[0].control
    last_w = window_stats(log, "herbs", 15, 20)
    factor = config.phases.shock_ticks / 5
    assert control.mean_price == last_w.mean_price
    assert control.traded_value == pytest.approx(last_w.traded_value * factor)
    assert control.gathered == round(last_w.gathered * factor)


def test_the_efficacy_check_runs_for_any_scenario(tmp_path: Path) -> None:
    from rpg_economy.analysis.shock_efficacy import efficacy_scenarios, measure_scenario

    config = make_config(
        phases={"base_ticks": 10, "shock_ticks": 10, "recovery_ticks": 10},
        time={"horizon_ticks": 30},
        metrics={"volatility_window_w": 3, "recovery_hold_ticks_m": 2},
    )
    scenarios = efficacy_scenarios(config)
    assert scenarios[0] is Scenario.RESOURCE_SCARCITY
    assert Scenario.RESOURCE_SCARCITY not in scenarios[1:]

    efficacy = measure_scenario(config, Scenario.DEMAND_SURGE, seeds=[1001], directory=tmp_path)
    assert efficacy.scenario == "demand_surge"
    assert efficacy.target == "health_potion"
    assert [effect.seed for effect in efficacy.effects] == [1001]


# ---------------------------------------------------------------------------
# --pilot-logs reads the pilot only, each seed once
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_experiment_2_logs_without_a_budget_are_left_out_not_fatal(tmp_path: Path) -> None:
    """Experiment 2 is not the pilot; a config without a budget cannot even derive its cell."""
    from rpg_economy.analysis.shock_efficacy import from_logs
    from rpg_economy.logging_ import run_logs

    budgeted = make_config(**SHORT, budget={"experiment_2_total_tokens": 20_000})
    _pilot_trio(budgeted, tmp_path / "exp2" / "1001", 1001, experiment=ExperimentRegime.EQUAL_COST)
    _pilot_trio(budgeted, tmp_path / "exp1" / "1001", 1001)

    config = make_config(**SHORT)
    groups, unmatched = from_logs(run_logs(tmp_path), config, allow_mock=True)
    assert unmatched == []
    assert set(groups) == {("resource_scarcity", c) for c in ("A", "B", "C")}
    # One seed, counted once: the Experiment 2 run of the same seed is not read.
    assert all([e.seed for e in g.effects] == [1001] for g in groups.values())


def test_a_candidate_that_cannot_be_derived_does_not_match(tmp_path: Path) -> None:
    """A ValueError from ``for_cell`` makes the log unmatched rather than crashing."""
    from rpg_economy.analysis.shock_efficacy import _matching_config
    from rpg_economy.logging_ import read_metadata

    budgeted = make_config(**SHORT, budget={"experiment_2_total_tokens": 20_000})
    runs = _pilot_trio(budgeted, tmp_path / "1001", 1001, experiment=ExperimentRegime.EQUAL_COST)
    metadata = read_metadata(next(iter(runs.values())).log_path)
    assert _matching_config(metadata, [make_config(**SHORT)]) is None
    assert _matching_config(metadata, [make_config(**SHORT), budgeted])[0] == 1


def test_only_pilot_seeds_of_the_pilot_phase_are_read(tmp_path: Path) -> None:
    from rpg_economy.analysis.shock_efficacy import from_logs
    from rpg_economy.logging_ import run_logs

    config = make_config(**SHORT)
    final_seed = config.seeds.final[0]
    _pilot_trio(config, tmp_path / str(final_seed), final_seed, phase="final")
    _pilot_trio(config, tmp_path / "adhoc" / "1002", 1002, phase="adhoc")
    _pilot_trio(config, tmp_path / "1001", 1001)

    groups, unmatched = from_logs(run_logs(tmp_path), config, allow_mock=True)
    assert unmatched == []
    assert {e.seed for g in groups.values() for e in g.effects} == {1001}


def test_mock_runs_are_left_out_unless_allowed(tmp_path: Path) -> None:
    from rpg_economy.analysis.shock_efficacy import from_logs
    from rpg_economy.logging_ import run_logs

    config = make_config(**SHORT)
    _pilot_trio(config, tmp_path / "1001", 1001)
    assert from_logs(run_logs(tmp_path), config) == ({}, [])
    assert from_logs(run_logs(tmp_path), config, allow_mock=True)[0]


def test_a_seed_run_twice_is_counted_once(tmp_path: Path) -> None:
    from rpg_economy.analysis.shock_efficacy import from_logs
    from rpg_economy.logging_ import run_logs

    config = make_config(**SHORT)
    _pilot_trio(config, tmp_path / "first" / "1001", 1001)
    _pilot_trio(config, tmp_path / "second" / "1001", 1001)
    groups, _ = from_logs(run_logs(tmp_path), config, allow_mock=True)
    assert all(len(g.effects) == 1 for g in groups.values())


def test_the_stockout_threshold_scales_with_a_compressed_shock_phase() -> None:
    """10 ticks of a full shock phase; in proportion when the phase is shorter."""
    from rpg_economy.analysis.shock_efficacy import Efficacy, SeedEffect, ShockPhase

    def efficacy(scale: float) -> Efficacy:
        control = ShockPhase(mean_price=10.0, stockout_ticks=0, traded_value=100.0)
        shocked = ShockPhase(mean_price=10.0, stockout_ticks=4, traded_value=100.0)
        return Efficacy(
            scenario="resource_scarcity",
            target="iron",
            effects=[SeedEffect(seed=1, shocked=shocked, control=control)],
            phase_scale=scale,
        )

    assert not efficacy(1.0).effective()
    assert efficacy(32 / 90).effective()
