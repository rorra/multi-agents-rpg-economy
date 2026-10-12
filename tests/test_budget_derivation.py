"""The Experiment 2 budget is derived from the whole pilot only (design v1.0 s8.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rpg_economy.analysis.budget_derivation import (
    BudgetDerivationError,
    derive_budget,
    find_logs,
)
from rpg_economy.domain import Condition
from rpg_economy.runner import run_trio
from tests.conftest import make_config
from tests.test_report_pipeline import (  # noqa: F401 -- recorded_phase is an autouse fixture
    record_phase,
    recorded_phase,
    with_phase,
)

SHORT = {
    "phases": {"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 4},
    "time": {"horizon_ticks": 12},
    "metrics": {"volatility_window_w": 2, "recovery_hold_ticks_m": 2},
    "seeds": {"pilot": [1001, 1002], "final": [2001, 2002]},
}


def _pilot(tmp_path: Path, seeds: tuple[int, ...], *, gate: bool = True, phase: str = "pilot"):
    """Pilot trios, each through the integrity gate the campaign runner applies."""
    config = make_config(**SHORT)
    runs = {}
    for seed in seeds:
        record_phase(tmp_path / str(seed), phase)
        runs[seed] = run_trio(with_phase(config, phase), tmp_path / str(seed), seed=seed)
    if gate:
        for seed in seeds:
            _pass_gate(tmp_path / str(seed))
    return config, runs


def _pass_gate(directory: Path) -> None:
    from rpg_economy.integrity import verify_trio_logs, write_integrity_marker

    write_integrity_marker(directory, verify_trio_logs(directory))


@pytest.mark.invariant
def test_a_pilot_missing_a_seed_is_refused_naming_it(tmp_path: Path) -> None:
    config, _ = _pilot(tmp_path, (1001,))
    with pytest.raises(BudgetDerivationError, match=r"pilot seeds \[1002\]"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


@pytest.mark.invariant
def test_a_pilot_seed_missing_a_condition_is_refused_naming_it(tmp_path: Path) -> None:
    config, runs = _pilot(tmp_path, (1001, 1002))
    runs[1002][Condition.B].log_path.unlink()
    # The trio's marker pinned three logs; with one gone it no longer holds.
    with pytest.raises(BudgetDerivationError, match=r"1002.*integrity gate"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


def test_an_empty_pilot_lists_every_missing_seed() -> None:
    config = make_config(**SHORT)
    with pytest.raises(BudgetDerivationError, match=r"\[1001, 1002\]"):
        derive_budget([], config=config, allow_mock=True)


def test_the_whole_pilot_is_admitted(tmp_path: Path) -> None:
    config, _ = _pilot(tmp_path, (1001, 1002))
    derivation = derive_budget(find_logs(tmp_path), config=config, allow_mock=True)
    assert sorted({run.seed for run in derivation.runs}) == [1001, 1002]
    assert len(derivation.runs) == 6


@pytest.mark.invariant
def test_a_partial_pilot_is_refused_as_incomplete(tmp_path: Path) -> None:
    """Whichever runs happened to finish are not the pre-registered pool (s8.4)."""
    config, _ = _pilot(tmp_path, (1002,))
    with pytest.raises(BudgetDerivationError, match=r"the pilot is incomplete.*\[1001\]"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


# ---------------------------------------------------------------------------
# Only verified trios, run by the current code, set the budget
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_trio_that_never_passed_the_integrity_gate_is_refused(tmp_path: Path) -> None:
    config, _ = _pilot(tmp_path, (1001, 1002), gate=False)
    _pass_gate(tmp_path / "1001")
    with pytest.raises(BudgetDerivationError, match=r"1002.*integrity gate"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


def test_a_log_changed_after_the_gate_is_refused(tmp_path: Path) -> None:
    config, runs = _pilot(tmp_path, (1001, 1002))
    log = runs[1001][Condition.A].log_path
    log.write_text(log.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BudgetDerivationError, match="integrity gate"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


@pytest.mark.invariant
def test_runs_of_other_prompts_are_refused(tmp_path: Path, monkeypatch) -> None:
    from rpg_economy.analysis import budget_derivation

    config, _ = _pilot(tmp_path, (1001, 1002))
    current = budget_derivation.current_prompt_hashes()
    monkeypatch.setattr(
        budget_derivation,
        "current_prompt_hashes",
        lambda: {**current, "version": current["version"] + "-next"},
    )
    with pytest.raises(BudgetDerivationError, match=r"prompts differ.*version"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


def test_runs_on_another_interpreter_are_refused(tmp_path: Path, monkeypatch) -> None:
    from rpg_economy.logging_ import metadata

    config, _ = _pilot(tmp_path, (1001, 1002))
    monkeypatch.setattr(metadata, "current_python_version", lambda: "3.14.999")
    with pytest.raises(BudgetDerivationError, match="Python"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


@pytest.mark.invariant
def test_runs_of_other_code_are_refused(tmp_path: Path, monkeypatch) -> None:
    """The prompts can match while the engine or accounting changed: the digest decides."""
    from rpg_economy.logging_ import metadata

    config, _ = _pilot(tmp_path, (1001, 1002))
    monkeypatch.setattr(metadata, "code_digest", lambda: "0" * 64)
    with pytest.raises(BudgetDerivationError, match=r"code digest [0-9a-f]{64} is not .*0{64}"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


def test_a_run_without_a_recorded_code_digest_is_refused(tmp_path: Path, monkeypatch) -> None:
    """A log written before the digest was recorded cannot show it ran this code."""
    from rpg_economy.analysis import budget_derivation
    from rpg_economy.logging_ import read_metadata

    config, _ = _pilot(tmp_path, (1001, 1002))
    original = budget_derivation.read_metadata
    monkeypatch.setattr(
        budget_derivation,
        "read_metadata",
        lambda path: original(path).model_copy(update={"code_digest": None}),
    )
    assert read_metadata(find_logs(tmp_path)[0]).code_digest is not None
    with pytest.raises(BudgetDerivationError, match="none recorded"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


def test_the_recorded_code_digest_is_the_current_one(tmp_path: Path) -> None:
    from rpg_economy.logging_ import code_digest, read_metadata

    _, runs = _pilot(tmp_path, (1001,), gate=False)
    for run in runs[1001].values():
        assert read_metadata(run.log_path).code_digest == code_digest()


def test_the_recorded_prompt_hashes_are_the_current_ones(tmp_path: Path) -> None:
    """The check compares like with like: what a run records is what the code hashes."""
    from rpg_economy.analysis.budget_derivation import current_prompt_hashes
    from rpg_economy.logging_ import read_metadata

    _, runs = _pilot(tmp_path, (1001,), gate=False)
    for run in runs[1001].values():
        assert read_metadata(run.log_path).prompt_hashes == current_prompt_hashes()


# ---------------------------------------------------------------------------
# Only finished pilot-phase runs set the budget
# ---------------------------------------------------------------------------
@pytest.mark.invariant
@pytest.mark.parametrize("phase", ["final", "adhoc"])
def test_runs_outside_the_pilot_phase_are_refused(tmp_path: Path, phase: str) -> None:
    config, _ = _pilot(tmp_path, (1001, 1002), phase=phase)
    with pytest.raises(BudgetDerivationError, match=f"phase is {phase}") as caught:
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)
    assert caught.value.key == "error.budget.not_pilot_phase"
    assert caught.value.params["phase"] == phase


def test_a_run_without_a_recorded_phase_is_refused(tmp_path: Path, monkeypatch) -> None:
    from rpg_economy.analysis import budget_derivation

    config, _ = _pilot(tmp_path, (1001, 1002))
    patched = budget_derivation.read_metadata
    monkeypatch.setattr(
        budget_derivation,
        "read_metadata",
        lambda path: patched(path).model_copy(update={"phase": None}),
    )
    with pytest.raises(BudgetDerivationError, match="phase is none"):
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)


@pytest.mark.invariant
def test_a_log_that_ends_in_run_aborted_cannot_set_the_budget(tmp_path: Path) -> None:
    import json

    config, runs = _pilot(tmp_path, (1001, 1002), gate=False)
    log = runs[1002][Condition.C].log_path
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    event = json.loads(lines[-1])
    event["type"] = "run_aborted"
    lines[-1] = json.dumps(event) + "\n"
    log.write_text("".join(lines), encoding="utf-8")
    # Gated as it stands, so only the aborted run itself can refuse it.
    for seed in (1001, 1002):
        _pass_gate(tmp_path / str(seed))
    with pytest.raises(BudgetDerivationError, match="did not finish") as caught:
        derive_budget(find_logs(tmp_path), config=config, allow_mock=True)
    assert caught.value.key == "error.budget.unfinished"


def test_a_budget_error_is_an_analysis_error_with_a_key() -> None:
    from rpg_economy.analysis.errors import AnalysisError

    config = make_config(**SHORT)
    with pytest.raises(AnalysisError) as caught:
        derive_budget([], config=config, allow_mock=True)
    assert caught.value.key == "error.budget.pilot_incomplete"
    assert caught.value.params == {"seeds": "1001, 1002", "scenario": "resource_scarcity"}


def test_a_real_run_with_an_unbilled_call_cannot_set_the_budget(tmp_path: Path) -> None:
    """A call priced from a local estimate is not the pilot's billed spend."""
    import json

    from rpg_economy.analysis.budget_derivation import BudgetDerivationError, _check_billed

    _, runs = _pilot(tmp_path, (1001,), gate=False)
    log = runs[1001][Condition.A].log_path
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    for k, line in enumerate(lines):
        event = json.loads(line)
        if event.get("type") == "tokens_consumed":
            total = event["payload"]["authoritative_total"]
            event["payload"]["billed_total"] = total - 1
            lines[k] = json.dumps(event) + "\n"
    log.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(BudgetDerivationError) as refused:
        _check_billed(log)
    assert refused.value.key == "error.budget.unbilled"
