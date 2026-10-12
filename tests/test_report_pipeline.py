"""From run logs to reports: the checks every log passes first (design s11, s12)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rpg_economy.analysis.pipeline import AnalysisError, analyse
from rpg_economy.cli import main
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition, ExperimentRegime, Scenario
from rpg_economy.i18n import clear_cache
from rpg_economy.runner import run_trio
from tests.conftest import ROOT, make_config

FINAL_SEED = 2001


@pytest.fixture(autouse=True)
def _reset_locale():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Run phase: which campaign stage made a log
# ---------------------------------------------------------------------------
#: Per trio directory, the phase its runs were made under, for logs whose
#: metadata does not record a phase (``RunMetadata.phase`` absent or unset).
_PHASES: dict[Path, str] = {}


def with_phase(config: SimulationConfig, phase: str) -> SimulationConfig:
    """``config`` set to run as ``phase``, where the configuration has a run phase."""
    if "phase" not in type(config.run).model_fields:
        return config
    payload = config.model_dump(mode="python")
    payload["run"] = {**payload["run"], "phase": phase}
    return SimulationConfig.model_validate(payload)


def record_phase(directory: Path, phase: str) -> None:
    _PHASES[Path(directory)] = phase


@pytest.fixture(autouse=True)
def recorded_phase(monkeypatch: pytest.MonkeyPatch):
    """A log reads as the phase its trio ran under, even where metadata cannot record it.

    Logs whose metadata records a phase are read as they are.
    """
    import rpg_economy.logging_ as logging_
    from rpg_economy.analysis import budget_derivation, pipeline

    original = logging_.read_metadata

    def read(path):
        metadata = original(path)
        phase = _PHASES.get(Path(path).parent)
        if getattr(metadata, "phase", None) is None and phase is not None:
            return metadata.model_copy(update={"phase": phase})
        return metadata

    for module in (logging_, pipeline, budget_derivation):
        monkeypatch.setattr(module, "read_metadata", read)
    yield
    _PHASES.clear()


def pass_gate(directory: Path) -> None:
    """The integrity gate the campaign runner applies to every trio."""
    from rpg_economy.integrity import verify_trio_logs, write_integrity_marker

    write_integrity_marker(directory, verify_trio_logs(directory))


LAYER = {
    "phases": {"base_ticks": 10, "shock_ticks": 10, "recovery_ticks": 20},
    "time": {"horizon_ticks": 40},
    "metrics": {"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
    "seeds": {"pilot": [1001], "final": [FINAL_SEED, 2002]},
}


def short_config(**overrides):
    return make_config(**{**LAYER, **overrides})


def run_cell(
    config,
    directory: Path,
    seed: int,
    experiment=ExperimentRegime.EQUAL_ACTIVITY,
    *,
    scenario=Scenario.RESOURCE_SCARCITY,
    phase: str | None = None,
    gate: bool = True,
):
    """One trio as a campaign makes it: in its phase, then through the integrity gate.

    The phase defaults to the one a campaign would run the seed in.
    """
    if phase is None:
        phase = "pilot" if seed in config.seeds.pilot else "final"
    cell = with_phase(config, phase).for_cell(scenario=scenario, experiment=experiment, seed=seed)
    trio = directory / str(seed)
    record_phase(trio, phase)
    runs = run_trio(cell, trio, seed=seed, cycles=config.time.total_cycles)
    if gate:
        pass_gate(trio)
    return runs


def test_a_mock_run_is_refused_unless_allowed(tmp_path: Path) -> None:
    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)

    with pytest.raises(AnalysisError, match="mock"):
        analyse(tmp_path, config)
    analysis = analyse(tmp_path, config, allow_mock=True)
    assert list(analysis.results) == [("resource_scarcity", "experiment_1")]


def test_an_unfinished_log_is_refused(tmp_path: Path) -> None:
    config = short_config()
    runs = run_cell(config, tmp_path, FINAL_SEED)
    log = runs[Condition.B].log_path
    lines = log.read_text().splitlines(keepends=True)
    log.write_text("".join(lines[:-1]))

    with pytest.raises(AnalysisError, match="did not finish"):
        analyse(tmp_path, config, allow_mock=True)


def test_a_log_from_another_configuration_is_refused(tmp_path: Path) -> None:
    run_cell(short_config(), tmp_path, FINAL_SEED)
    other = short_config(market={"order_expiry_ticks": 7})

    with pytest.raises(AnalysisError, match="different configuration"):
        analyse(tmp_path, other, allow_mock=True)


def test_pilot_seeds_are_left_out_of_the_inference(tmp_path: Path) -> None:
    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path, 1001)

    analysis = analyse(tmp_path, config, allow_mock=True)
    assert analysis.skipped_seeds == [1001]
    assert analysis.results[("resource_scarcity", "experiment_1")].seeds == [FINAL_SEED]

    included = analyse(tmp_path, config, allow_mock=True, include_pilot=True)
    assert included.results[("resource_scarcity", "experiment_1")].seeds == [1001, FINAL_SEED]


def test_a_cell_missing_a_condition_is_refused(tmp_path: Path) -> None:
    config = short_config()
    runs = run_cell(config, tmp_path, FINAL_SEED)
    runs[Condition.C].log_path.unlink()

    with pytest.raises(AnalysisError, match="need A, B and C"), pytest.warns(UserWarning):
        analyse(tmp_path, config, allow_mock=True)


def test_experiments_are_reported_separately_with_the_common_horizon(tmp_path: Path) -> None:
    config = short_config(budget={"experiment_2_total_tokens": 20_000})
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path / "exp2", FINAL_SEED, experiment=ExperimentRegime.EQUAL_COST)

    analysis = analyse(tmp_path, config, allow_mock=True)
    assert set(analysis.results) == {
        ("resource_scarcity", "experiment_1"),
        ("resource_scarcity", "experiment_2"),
    }
    exp1 = analysis.results[("resource_scarcity", "experiment_1")]
    exp2 = analysis.results[("resource_scarcity", "experiment_2")]
    assert all(run.common_horizon is None for run in exp1.runs)
    assert all(run.common_horizon is not None for run in exp2.runs)
    horizons = {run.common_horizon["horizon"] for run in exp2.runs}
    assert len(horizons) == 1  # one horizon per trio
    # Every economic metric is recomputed there, not only the primary ones.
    payload = exp2.to_payload()
    for metric in ("useful_production_common", "gini_common", "volume_all_goods_common"):
        assert metric in payload["contrasts"], metric
    assert "survival_propagation_common_horizon" in payload


def test_the_report_command_writes_both_languages(tmp_path: Path, monkeypatch) -> None:
    config = short_config()
    run_cell(config, tmp_path / "logs", FINAL_SEED)
    layer = tmp_path / "layer.yaml"
    layer.write_text(json.dumps(LAYER))
    out = tmp_path / "reports"
    code = main(
        ["report", str(tmp_path / "logs"), "--layer", str(layer), "--out", str(out), "--allow-mock"]
    )
    assert code == 0
    cell = out / "resource_scarcity-experiment_1"
    assert (cell / "en" / "report.txt").exists()
    assert (cell / "es" / "report.txt").exists()
    assert json.loads((cell / "en" / "results.json").read_text()) == json.loads(
        (cell / "es" / "results.json").read_text()
    )


def test_the_report_command_refuses_mock_runs_by_default(tmp_path: Path, capsys) -> None:
    config = short_config()
    run_cell(config, tmp_path / "logs", FINAL_SEED)
    layer = tmp_path / "layer.yaml"
    layer.write_text(json.dumps(LAYER))
    code = main(["report", str(tmp_path / "logs"), "--layer", str(layer), "--out", str(tmp_path)])
    assert code == 2
    assert "mock" in capsys.readouterr().err


def test_a_seed_without_a_complete_trio_is_left_out_with_a_warning(tmp_path: Path) -> None:
    from rpg_economy.analysis.pipeline import IncompleteTrioWarning

    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    runs = run_cell(config, tmp_path, 2002)
    runs[Condition.C].log_path.unlink()

    with pytest.warns(IncompleteTrioWarning, match="2002"):
        analysis = analyse(tmp_path, config, allow_mock=True)
    key = ("resource_scarcity", "experiment_1")
    assert analysis.incomplete_trios == {key: [2002]}
    results = analysis.results[key]
    assert results.seeds == [FINAL_SEED]
    assert {run.condition for run in results.runs} == {"A", "B", "C"}
    # The results file names the dropped seeds too, not only the console (s11.3).
    assert results.incomplete_trios == [2002]
    assert results.to_payload()["incomplete_trios"] == [2002]


def test_an_incomplete_trio_does_not_set_the_common_horizon(tmp_path: Path) -> None:
    config = short_config(budget={"experiment_2_total_tokens": 20_000})
    run_cell(config, tmp_path, FINAL_SEED, experiment=ExperimentRegime.EQUAL_COST)
    runs = run_cell(config, tmp_path, 2002, experiment=ExperimentRegime.EQUAL_COST)
    runs[Condition.A].log_path.unlink()

    with pytest.warns(UserWarning):
        analysis = analyse(tmp_path, config, allow_mock=True)
    results = analysis.results[("resource_scarcity", "experiment_2")]
    assert results.seeds == [FINAL_SEED]
    assert all(run.common_horizon is not None for run in results.runs)


def test_a_report_on_pilot_seeds_is_flagged(tmp_path: Path) -> None:
    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path, 1001)
    key = ("resource_scarcity", "experiment_1")

    assert not analyse(tmp_path, config, allow_mock=True).results[key].pilot_seeds_included
    included = analyse(tmp_path, config, allow_mock=True, include_pilot=True).results[key]
    assert included.pilot_seeds_included
    assert included.to_payload()["pilot_seeds_included"] is True


def test_a_cell_without_any_complete_trio_is_skipped_not_fatal(tmp_path: Path) -> None:
    """One bad cell must not cost the report of every other cell."""
    from rpg_economy.analysis.pipeline import IncompleteTrioWarning

    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    runs = run_cell(config, tmp_path / "surge", FINAL_SEED, scenario=Scenario.DEMAND_SURGE)
    runs[Condition.C].log_path.unlink()

    with pytest.warns(IncompleteTrioWarning, match="demand_surge"):
        analysis = analyse(tmp_path, config, allow_mock=True)
    assert analysis.skipped_cells == {("demand_surge", "experiment_1"): ["A", "B"]}
    assert list(analysis.results) == [("resource_scarcity", "experiment_1")]
    results = analysis.results[("resource_scarcity", "experiment_1")]
    assert results.seeds == [FINAL_SEED]
    assert results.to_payload()["skipped_cells"] == {"demand_surge/experiment_1": ["A", "B"]}
    assert results.incomplete_trios == []


# ---------------------------------------------------------------------------
# The integrity gate: a trio that fails the check is rejected (design s12)
# ---------------------------------------------------------------------------
KEY = ("resource_scarcity", "experiment_1")


@pytest.mark.invariant
def test_a_trio_that_never_passed_the_gate_is_rejected_and_reported(tmp_path: Path) -> None:
    from rpg_economy.analysis.pipeline import RejectedTrioWarning

    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path, 2002, gate=False)

    with pytest.warns(RejectedTrioWarning, match="2002"):
        analysis = analyse(tmp_path, config, allow_mock=True)
    results = analysis.results[KEY]
    assert results.seeds == [FINAL_SEED]
    assert [(e["seed"], e["reason"]) for e in analysis.rejected_trios[KEY]] == [
        (2002, "integrity_marker")
    ]
    payload = results.to_payload()
    assert [e["seed"] for e in payload["rejected_trios"]["resource_scarcity/experiment_1"]] == [
        2002
    ]


def test_a_log_changed_after_the_gate_rejects_its_trio(tmp_path: Path) -> None:
    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    runs = run_cell(config, tmp_path, 2002)
    log = runs[Condition.A].log_path
    log.write_text(log.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="integrity"):
        analysis = analyse(tmp_path, config, allow_mock=True)
    assert analysis.results[KEY].seeds == [FINAL_SEED]
    assert analysis.rejected_trios[KEY][0]["reason"] == "integrity_marker"


@pytest.mark.invariant
def test_a_marker_horizon_that_disagrees_with_the_logs_refuses_the_trio(tmp_path: Path) -> None:
    from rpg_economy.integrity import INTEGRITY_MARKER

    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path, 2002)
    marker = tmp_path / "2002" / INTEGRITY_MARKER
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["horizon"]["common_active_horizon"] == config.time.horizon_ticks
    payload["horizon"]["common_active_horizon"] = 17
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.warns(UserWarning, match="horizon_mismatch"):
        analysis = analyse(tmp_path, config, allow_mock=True)
    assert analysis.results[KEY].seeds == [FINAL_SEED]
    (entry,) = analysis.rejected_trios[KEY]
    assert (entry["seed"], entry["reason"]) == (2002, "horizon_mismatch")
    assert (entry["recorded"], entry["computed"]) == (17, config.time.horizon_ticks)


def test_an_experiment_2_marker_agrees_with_the_computed_horizon(tmp_path: Path) -> None:
    config = short_config(budget={"experiment_2_total_tokens": 20_000})
    run_cell(config, tmp_path, FINAL_SEED, experiment=ExperimentRegime.EQUAL_COST)
    analysis = analyse(tmp_path, config, allow_mock=True)
    assert analysis.rejected_trios == {}
    assert analysis.results[("resource_scarcity", "experiment_2")].seeds == [FINAL_SEED]


def test_every_trio_rejected_refuses_the_analysis(tmp_path: Path) -> None:
    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED, gate=False)
    with pytest.raises(AnalysisError) as caught, pytest.warns(UserWarning):
        analyse(tmp_path, config, allow_mock=True)
    assert caught.value.key == "error.report.no_complete_trio"
    assert caught.value.params["rejected"] == 1


# ---------------------------------------------------------------------------
# Run phase: only final runs enter the inference (user ruling)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_only_final_phase_logs_enter_the_inference(tmp_path: Path) -> None:
    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path / "adhoc", 2002, phase="adhoc")
    run_cell(config, tmp_path / "pilot", 2002, phase="pilot")

    analysis = analyse(tmp_path, config, allow_mock=True)
    assert analysis.results[KEY].seeds == [FINAL_SEED]
    reasons = sorted((e["seed"], e["phase"], e["reason"]) for e in analysis.skipped_logs)
    assert reasons == [(2002, "adhoc", "phase_adhoc")] * 3 + [(2002, "pilot", "phase_pilot")] * 3
    assert analysis.skipped_seeds == []
    assert len(analysis.results[KEY].to_payload()["skipped_logs"]) == 6


def test_include_pilot_admits_pilot_phase_runs_too(tmp_path: Path) -> None:
    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path, 2002, phase="pilot")
    run_cell(config, tmp_path, 1001)

    analysis = analyse(tmp_path, config, allow_mock=True, include_pilot=True)
    assert analysis.results[KEY].seeds == [1001, FINAL_SEED, 2002]
    assert analysis.skipped_logs == []


def test_a_log_without_a_recorded_phase_is_left_out(tmp_path: Path, monkeypatch) -> None:
    from rpg_economy.analysis import pipeline

    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path, 2002)
    patched = pipeline.read_metadata

    def older(path):
        metadata = patched(path)
        if metadata.seed == 2002:
            return metadata.model_copy(update={"phase": None})
        return metadata

    monkeypatch.setattr(pipeline, "read_metadata", older)
    analysis = analyse(tmp_path, config, allow_mock=True)
    assert analysis.results[KEY].seeds == [FINAL_SEED]
    assert {e["reason"] for e in analysis.skipped_logs} == {"phase_missing"}


def test_no_admitted_log_is_refused_with_its_reasons(tmp_path: Path) -> None:
    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED, phase="adhoc")
    with pytest.raises(AnalysisError, match="phase_adhoc") as caught:
        analyse(tmp_path, config, allow_mock=True)
    assert caught.value.key == "error.report.no_admitted_logs"


@pytest.mark.invariant
def test_a_stray_mock_log_outside_the_analysis_does_not_abort_it(
    tmp_path: Path, monkeypatch
) -> None:
    """Selection first, then the mock check: a stray smoke run is left out, not fatal."""
    from rpg_economy.analysis import pipeline

    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path / "smoke", 2002, phase="adhoc")
    run_cell(config, tmp_path / "pilot", 1001)
    patched = pipeline.read_metadata

    def real_final(path):
        # The final trio stands for real-model runs; the others stay mock.
        metadata = patched(path)
        if metadata.seed == FINAL_SEED:
            return metadata.model_copy(update={"llm_mock": False})
        return metadata

    monkeypatch.setattr(pipeline, "read_metadata", real_final)
    analysis = analyse(tmp_path, config)
    assert analysis.results[KEY].seeds == [FINAL_SEED]
    left_out = {(e["seed"], e["reason"], e["mock"]) for e in analysis.skipped_logs}
    assert left_out == {(2002, "phase_adhoc", True), (1001, "seed_not_analysed", True)}


# ---------------------------------------------------------------------------
# Every admitted log: finished, at the full horizon (as derive-budget checks)
# ---------------------------------------------------------------------------
def _rewrite_last_event(path: Path, **changes) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    event = json.loads(lines[-1])
    for name, value in changes.items():
        if name == "ticks":
            event["payload"]["ticks"] = value
        else:
            event[name] = value
    lines[-1] = json.dumps(event) + "\n"
    path.write_text("".join(lines), encoding="utf-8")


@pytest.mark.invariant
def test_a_log_that_ends_in_run_aborted_is_refused(tmp_path: Path) -> None:
    config = short_config()
    runs = run_cell(config, tmp_path, FINAL_SEED)
    _rewrite_last_event(runs[Condition.B].log_path, type="run_aborted")

    with pytest.raises(AnalysisError, match="did not finish") as caught:
        analyse(tmp_path, config, allow_mock=True)
    assert caught.value.key == "error.report.unfinished"
    assert caught.value.params == {"path": str(runs[Condition.B].log_path)}


@pytest.mark.invariant
def test_a_log_short_of_the_horizon_is_refused(tmp_path: Path) -> None:
    config = short_config()
    runs = run_cell(config, tmp_path, FINAL_SEED)
    _rewrite_last_event(runs[Condition.A].log_path, ticks=config.time.horizon_ticks - 1)

    with pytest.raises(AnalysisError, match="not the full") as caught:
        analyse(tmp_path, config, allow_mock=True)
    assert caught.value.key == "error.report.short_horizon"
    assert caught.value.params["horizon"] == config.time.horizon_ticks


def test_collect_cells_admits_pilot_logs_after_the_config_gains_a_budget_or_seeds(
    tmp_path: Path,
) -> None:
    """Deriving the budget, or extending the seed lists, leaves experiment_1 runs as they were."""
    from rpg_economy.analysis.pipeline import collect_cells
    from rpg_economy.logging_ import run_logs

    run_cell(short_config(), tmp_path, 1001)
    later = short_config(
        budget={"experiment_2_total_tokens": 20_000},
        seeds={"pilot": [1001, 1002], "final": [FINAL_SEED, 2002, 2003]},
    )
    collected = collect_cells(run_logs(tmp_path), later, include_pilot=True, allow_mock=True)
    assert set(collected.cells[KEY].logs) == {(c, 1001) for c in ("A", "B", "C")}
    assert collected.skipped_logs == []


# ---------------------------------------------------------------------------
# A cell whose runs mix code versions is refused (design s12)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("code_digest", "0" * 64),
        ("prompt_hashes", {"version": "other"}),
        ("python_version", "3.14.999"),
        ("global_view_version", "other"),
    ],
)
def test_a_cell_mixing_code_versions_is_refused_naming_the_field(
    tmp_path: Path, monkeypatch, field: str, value
) -> None:
    from rpg_economy.analysis import pipeline

    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path, 2002)
    patched = pipeline.read_metadata

    def other_version(path):
        metadata = patched(path)
        if metadata.seed == 2002:
            return metadata.model_copy(update={field: value})
        return metadata

    monkeypatch.setattr(pipeline, "read_metadata", other_version)
    with pytest.raises(AnalysisError) as caught, pytest.warns(UserWarning, match=field):
        analyse(tmp_path, config, allow_mock=True)
    assert caught.value.key == "error.report.no_complete_trio"
    assert caught.value.params["refused"] == 1


def test_a_refused_cell_is_named_in_the_other_cells_results(tmp_path: Path, monkeypatch) -> None:
    from rpg_economy.analysis import pipeline

    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    run_cell(config, tmp_path / "surge", FINAL_SEED, scenario=Scenario.DEMAND_SURGE)
    run_cell(config, tmp_path / "surge", 2002, scenario=Scenario.DEMAND_SURGE)
    patched = pipeline.read_metadata

    def other_code(path):
        metadata = patched(path)
        if metadata.scenario is Scenario.DEMAND_SURGE and metadata.seed == 2002:
            return metadata.model_copy(update={"code_digest": "0" * 64})
        return metadata

    monkeypatch.setattr(pipeline, "read_metadata", other_code)
    with pytest.warns(UserWarning, match="code_digest"):
        analysis = analyse(tmp_path, config, allow_mock=True)
    assert analysis.refused_cells == {("demand_surge", "experiment_1"): ["code_digest"]}
    assert list(analysis.results) == [KEY]
    payload = analysis.results[KEY].to_payload()
    assert payload["refused_cells"] == {"demand_surge/experiment_1": ["code_digest"]}


# ---------------------------------------------------------------------------
# The metric parameters the analysis applied are recorded (user ruling)
# ---------------------------------------------------------------------------
def test_the_results_record_the_metric_parameters_applied(tmp_path: Path) -> None:
    config = short_config()
    run_cell(config, tmp_path, FINAL_SEED)
    results = analyse(tmp_path, config, allow_mock=True).results[KEY]
    parameters = results.to_payload()["metric_parameters"]
    assert parameters == config.metrics.model_dump(mode="json")
    for name in (
        "volatility_window_w",
        "recovery_band_pct_x",
        "recovery_hold_ticks_m",
        "useful_production_ticks_t",
        "unanswered_request_ticks_d",
    ):
        assert name in parameters


# ---------------------------------------------------------------------------
# Every refusal names a catalog message, in both languages
# ---------------------------------------------------------------------------
def test_every_analysis_error_carries_a_catalog_key() -> None:
    sources = [
        ROOT / "src" / "rpg_economy" / "analysis" / name
        for name in ("pipeline.py", "budget_derivation.py")
    ]
    raised = 0
    keys: set[str] = set()
    for source in sources:
        text = source.read_text(encoding="utf-8")
        raised += len(re.findall(r"raise (?:AnalysisError|BudgetDerivationError)\(", text))
        keys |= set(re.findall(r'key="(error\.[a-z_.]+)"', text))
    assert raised == sum(
        len(re.findall(r'key="error\.', source.read_text(encoding="utf-8"))) for source in sources
    )
    for locale in ("en", "es"):
        catalog = json.loads((ROOT / "locales" / f"{locale}.json").read_text(encoding="utf-8"))
        missing = sorted(key for key in keys if key not in catalog)
        assert not missing, f"{locale}: {missing}"


def test_an_analysis_error_keeps_its_english_message() -> None:
    error = AnalysisError("x: the run did not finish", key="error.report.unfinished", path="x")
    assert str(error) == "x: the run did not finish"
    assert (error.key, error.params) == ("error.report.unfinished", {"path": "x"})
