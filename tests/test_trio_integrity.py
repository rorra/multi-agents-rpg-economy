"""The trio structural-sanity gate (design v1.0 s12).

The gate rejects a trio whose three conditions ran identically while presenting
as valid. These tests confirm the gate detects each way that can happen -- a
gate that cannot fail would protect nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition, TokenBucket
from rpg_economy.integrity import TrioIntegrityError, verify_trio
from rpg_economy.llm import TokenLedger
from rpg_economy.runner import run_condition, run_trio
from tests.conftest import make_config


@dataclass
class FakeRun:
    """Enough of a RunResult for the gate to inspect."""

    ledger: TokenLedger
    log_path: Path


def ledger_with(**buckets: int) -> TokenLedger:
    ledger = TokenLedger()
    for name, tokens in buckets.items():
        ledger.record(
            input_by_bucket={TokenBucket(name): tokens},
            output_tokens=0,
            output_bucket=TokenBucket(name),
        )
    return ledger


def healthy_trio(tmp_path: Path) -> dict[Condition, FakeRun]:
    """A trio shaped exactly as the design says it must be."""
    runs = {}
    for condition, buckets in (
        (Condition.A, {"agent": 100, "director": 50}),
        (Condition.B, {"agent": 100, "p2p_content": 10}),
        (Condition.C, {"agent": 100, "p2p_content": 10, "bulletin": 20}),
    ):
        path = tmp_path / f"{condition.value}.jsonl"
        path.write_text(
            f'{{"seq":0,"tick":0,"cycle":0,"type":"tick","agent_id":null,'
            f'"payload":{{"c":"{condition.value}"}}}}\n',
            encoding="utf-8",
        )
        runs[condition] = FakeRun(ledger=ledger_with(**buckets), log_path=path)
    return runs


def test_a_healthy_trio_passes(tmp_path: Path) -> None:
    report = verify_trio(healthy_trio(tmp_path))
    assert "director_tokens_present_in_a" in report.checks_passed
    assert "logs_pairwise_distinct" in report.checks_passed
    # Hand-built logs carry no run metadata, so only the parity check, which
    # reads it, is skipped.
    assert report.checks_skipped == ["metadata_parity"]


@pytest.mark.invariant
def test_the_gate_catches_a_condition_that_never_took_effect(tmp_path: Path) -> None:
    """A ran without a Director."""
    runs = healthy_trio(tmp_path)
    runs[Condition.A] = FakeRun(ledger=ledger_with(agent=100), log_path=runs[Condition.A].log_path)

    with pytest.raises(TrioIntegrityError, match="no Director tokens"):
        verify_trio(runs)


@pytest.mark.invariant
def test_the_gate_catches_identical_logs(tmp_path: Path) -> None:
    """Three conditions producing the same bytes did not run as three conditions."""
    runs = healthy_trio(tmp_path)
    shared = runs[Condition.B].log_path.read_text(encoding="utf-8")
    runs[Condition.C].log_path.write_text(shared, encoding="utf-8")

    with pytest.raises(TrioIntegrityError, match="byte-identical"):
        verify_trio(runs)


@pytest.mark.invariant
def test_the_gate_catches_a_director_outside_condition_a(tmp_path: Path) -> None:
    runs = healthy_trio(tmp_path)
    runs[Condition.B] = FakeRun(
        ledger=ledger_with(agent=100, director=5), log_path=runs[Condition.B].log_path
    )
    with pytest.raises(TrioIntegrityError, match="no coordinator"):
        verify_trio(runs)


@pytest.mark.invariant
def test_the_gate_catches_peer_traffic_in_condition_a(tmp_path: Path) -> None:
    runs = healthy_trio(tmp_path)
    runs[Condition.A] = FakeRun(
        ledger=ledger_with(agent=100, director=50, p2p_content=7),
        log_path=runs[Condition.A].log_path,
    )
    with pytest.raises(TrioIntegrityError, match="p2p_content"):
        verify_trio(runs)


@pytest.mark.invariant
def test_the_gate_catches_a_bulletin_outside_condition_c(tmp_path: Path) -> None:
    runs = healthy_trio(tmp_path)
    runs[Condition.B] = FakeRun(
        ledger=ledger_with(agent=100, bulletin=9), log_path=runs[Condition.B].log_path
    )
    with pytest.raises(TrioIntegrityError, match="bulletin tokens"):
        verify_trio(runs)


@pytest.mark.invariant
def test_the_gate_catches_condition_c_without_its_ablation(tmp_path: Path) -> None:
    """C without a bulletin is indistinguishable from B."""
    runs = healthy_trio(tmp_path)
    runs[Condition.C] = FakeRun(
        ledger=ledger_with(agent=100, p2p_content=10), log_path=runs[Condition.C].log_path
    )
    with pytest.raises(TrioIntegrityError, match="no bulletin tokens"):
        verify_trio(runs)


def test_the_gate_reports_what_it_could_not_verify(tmp_path: Path) -> None:
    """A trio too short to exercise a check must not pass as though it had."""
    runs = healthy_trio(tmp_path)
    for condition in (Condition.A, Condition.C):
        runs[condition] = FakeRun(ledger=TokenLedger(), log_path=runs[condition].log_path)
    report = verify_trio(runs)

    assert "director_tokens_present_in_a" in report.checks_skipped
    assert "bulletin_tokens_present_in_c" in report.checks_skipped


def test_a_missing_condition_is_caught(tmp_path: Path) -> None:
    runs = healthy_trio(tmp_path)
    del runs[Condition.C]
    with pytest.raises(TrioIntegrityError, match="missing conditions"):
        verify_trio(runs)


# ---------------------------------------------------------------------------
# The condition field itself
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_string_condition_is_rejected_at_the_point_of_use(tmp_path: Path) -> None:
    """model_copy does not validate; the run must refuse rather than mis-run."""
    config = make_config()
    broken = config.model_copy(update={"run": config.run.model_copy(update={"condition": "A"})})
    with pytest.raises(TypeError, match="not a Condition enum member"):
        run_condition(broken, tmp_path / "run.jsonl", cycles=1)


@pytest.mark.invariant
def test_for_condition_rejects_a_string() -> None:
    with pytest.raises(TypeError, match="Condition enum member"):
        make_config().for_condition("A")


def test_for_condition_produces_a_validated_config() -> None:
    config = make_config().for_condition(Condition.C, seed=4242)
    assert config.run.condition is Condition.C
    assert config.run.seed == 4242
    assert isinstance(config, SimulationConfig)


def test_for_condition_changes_nothing_else() -> None:
    base = make_config()
    derived = base.for_condition(Condition.C)
    left, right = base.model_dump(mode="json"), derived.model_dump(mode="json")
    assert {k for k in left if left[k] != right[k]} == {"run"}


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_run_trio_verifies_every_trio_it_returns(tmp_path: Path) -> None:
    """The gate is enforced by the runner, not only by the test suite."""
    import inspect

    from rpg_economy import runner

    source = inspect.getsource(runner.run_trio)
    assert "verify_trio(runs)" in source

    runs = run_trio(make_config(), tmp_path, seed=1001, cycles=8)
    assert verify_trio(runs).checks_passed
