"""Structural sanity checks on a seed-paired trio (design v1.0 s12).

These assert properties that are definitional rather than empirical: they
follow from what conditions A, B and C *are*, so they can never be a legitimate
research finding and their violation always means a defect.

A trio whose three runs silently executed as the same condition (for example, a
condition override passed as a string rather than an enum) would still produce
numbers, and every downstream contrast would report no difference between
architectures. Nothing in the output would look wrong, so this runs inside the
runner, on every trio, not only in the test suite.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpg_economy.domain import Condition, TokenBucket
from rpg_economy.logging_ import EventType, LogError, log_digest, read_events, read_metadata

#: The file a trio directory gets once its three logs have passed the gate. A
#: file rather than a log event: each run log is append-only and belongs to one
#: run, while the verdict belongs to the trio.
INTEGRITY_MARKER = "integrity.json"

#: Metadata fields the three runs of a trio must share. The same package
#: source, prompt templates and interpreter, the same setup, and the same
#: executor (real provider or offline mock, which the setup hash leaves out): a
#: trio assembled from runs of different code, setups or executors compares
#: those, not architectures (design v1.0 s12).
PARITY_FIELDS = ("code_digest", "llm_mock", "prompt_hashes", "python_version", "setup_hash")


class TrioIntegrityError(AssertionError):
    """Raised when a trio is not structurally three distinct conditions."""


@dataclass
class TrioIntegrityReport:
    """What was checked, and what could be checked."""

    checks_passed: list[str] = field(default_factory=list)
    checks_skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return True

    def to_payload(self) -> dict:
        return {
            "passed": list(self.checks_passed),
            "skipped": list(self.checks_skipped),
        }


def verify_trio(runs: dict[Condition, object]) -> TrioIntegrityReport:
    """Verify that a trio really is three different conditions.

    Raises ``TrioIntegrityError`` on the first violation. Positive checks that
    need activity are skipped -- and recorded as skipped -- when a run made no
    agent calls, so a trio too short to exercise them cannot pass silently as
    though it had.
    """
    missing = set(Condition) - set(runs)
    if missing:
        raise TrioIntegrityError(f"trio is missing conditions: {sorted(c.value for c in missing)}")

    report = TrioIntegrityReport()

    def active(condition: Condition) -> bool:
        return runs[condition].ledger.calls > 0

    def tokens(condition: Condition, bucket: TokenBucket) -> int:
        return runs[condition].ledger.total(bucket)

    # -- the Director exists in A and only in A ----------------------------
    if active(Condition.A):
        if tokens(Condition.A, TokenBucket.DIRECTOR) <= 0:
            raise TrioIntegrityError(
                "Condition A consumed no Director tokens: the run was not centralized. "
                "This is the signature of a condition that never took effect."
            )
        report.checks_passed.append("director_tokens_present_in_a")
    else:
        report.checks_skipped.append("director_tokens_present_in_a")

    for condition in (Condition.B, Condition.C):
        if tokens(condition, TokenBucket.DIRECTOR) != 0:
            raise TrioIntegrityError(
                f"Condition {condition.value} consumed Director tokens; B and C have "
                f"no coordinator (design v1.0 s3.2, s3.3)"
            )
    report.checks_passed.append("no_director_tokens_in_b_and_c")

    # -- no peer traffic in A (design v1.0 s3.1) ---------------------------
    if tokens(Condition.A, TokenBucket.P2P_CONTENT) != 0:
        raise TrioIntegrityError(
            "Condition A consumed p2p_content tokens; economic agents send no "
            "peer-to-peer messages in A (design v1.0 s3.1)"
        )
    report.checks_passed.append("no_p2p_tokens_in_a")

    # -- the bulletin is C's and only C's ----------------------------------
    for condition in (Condition.A, Condition.B):
        if tokens(condition, TokenBucket.BULLETIN) != 0:
            raise TrioIntegrityError(
                f"Condition {condition.value} consumed bulletin tokens; only C "
                f"receives the bulletin (design v1.0 s3.3)"
            )
    report.checks_passed.append("no_bulletin_tokens_in_a_and_b")

    if active(Condition.C):
        if tokens(Condition.C, TokenBucket.BULLETIN) <= 0:
            raise TrioIntegrityError(
                "Condition C consumed no bulletin tokens: its ablation never took "
                "effect, so C is indistinguishable from B (design v1.0 s3.3)"
            )
        report.checks_passed.append("bulletin_tokens_present_in_c")
    else:
        report.checks_skipped.append("bulletin_tokens_present_in_c")

    # -- the three runs are actually different -----------------------------
    digests = {condition: log_digest(run.log_path) for condition, run in runs.items()}
    for left, right in (
        (Condition.A, Condition.B),
        (Condition.A, Condition.C),
        (Condition.B, Condition.C),
    ):
        if digests[left] == digests[right]:
            raise TrioIntegrityError(
                f"Conditions {left.value} and {right.value} produced byte-identical "
                f"logs: they did not run as different conditions"
            )
    report.checks_passed.append("logs_pairwise_distinct")

    # -- the three runs were written by the same code, for the same setup --
    _check_parity(runs, report)

    return report


def _check_parity(runs: dict[Condition, object], report: TrioIntegrityReport) -> None:
    """A, B and C share ``PARITY_FIELDS`` in their metadata.

    Skipped, and recorded as skipped, only when none of the three logs carries
    metadata (hand-built fixtures); a trio where some do and some do not is a
    defect like any other mismatch.
    """
    metadata = {}
    for condition, run in runs.items():
        try:
            metadata[condition] = read_metadata(run.log_path)
        except LogError, OSError:
            continue
    if not metadata:
        report.checks_skipped.append("metadata_parity")
        return
    if len(metadata) != len(runs):
        missing = sorted(c.value for c in runs if c not in metadata)
        raise TrioIntegrityError(f"conditions {missing} have no readable run metadata")
    for name in PARITY_FIELDS:
        values = {condition: getattr(meta, name) for condition, meta in metadata.items()}
        reference = values[Condition.A]
        different = sorted(c.value for c, value in values.items() if value != reference)
        if different:
            raise TrioIntegrityError(
                f"conditions {different} differ from A in {name}: the trio was not "
                f"run by the same code on the same setup"
            )
    report.checks_passed.append("metadata_parity")


# ---------------------------------------------------------------------------
# Verifying from logs, and recording the verdict
# ---------------------------------------------------------------------------
@dataclass
class LoggedLedger:
    """The token totals a finished run recorded in its ``tokens_consumed`` event."""

    calls: int
    totals: dict[str, int]

    def total(self, bucket: TokenBucket | None = None) -> int:
        if bucket is None:
            return sum(self.totals.values())
        return int(self.totals.get(bucket.value, 0))


@dataclass
class LoggedRun:
    """Enough of a run, rebuilt from its log, for ``verify_trio`` to inspect.

    Lets a trio be verified when some of its runs finished in an earlier
    attempt, so a condition that already ran is never run (and paid for) again
    just to be checked.
    """

    ledger: LoggedLedger
    log_path: Path


def run_from_log(path: Path | str) -> LoggedRun:
    """Rebuild a finished run's ledger totals from its log."""
    consumed: dict[str, Any] | None = None
    for event in read_events(path):
        if event.type is EventType.TOKENS_CONSUMED:
            consumed = event.payload
    if consumed is None:
        raise TrioIntegrityError(f"{path} has no tokens_consumed event; the run never finished")
    calls = consumed.get("calls", {})
    return LoggedRun(
        ledger=LoggedLedger(
            calls=sum(int(value) for value in calls.values()),
            totals={key: int(value) for key, value in consumed.get("total", {}).items()},
        ),
        log_path=Path(path),
    )


def verify_trio_logs(directory: Path | str) -> TrioIntegrityReport:
    """``verify_trio`` on the three finished logs of a trio directory."""
    directory = Path(directory)
    runs: dict[Condition, object] = {}
    for condition in (Condition.A, Condition.B, Condition.C):
        path = directory / f"{condition.value}.jsonl"
        if path.exists():
            runs[condition] = run_from_log(path)
    return verify_trio(runs)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_horizons(directory: Path | str) -> dict[str, Any]:
    """The trio's common active horizon, read from its three finished logs (design s8.4, s15).

    Per condition: the tick its token budget ran out (None if it never did)
    and the mechanical ticks that ran after it. The common active horizon is
    the earliest exhaustion tick among A, B and C, or the planned horizon --
    the ticks the runs reached -- if none exhausted.
    """
    from rpg_economy.llm.budget import common_active_horizon

    directory = Path(directory)
    conditions: dict[str, dict[str, int | None]] = {}
    planned: int | None = None
    for condition in (Condition.A, Condition.B, Condition.C):
        exhausted: int | None = None
        after = 0
        ticks: int | None = None
        for event in read_events(directory / f"{condition.value}.jsonl"):
            if event.type is EventType.BUDGET_EXHAUSTED and exhausted is None:
                exhausted = event.tick
            elif event.type is EventType.TICK and exhausted is not None:
                after += 1
            elif event.type is EventType.RUN_END:
                ticks = event.payload.get("ticks")
        conditions[condition.value] = {
            "exhaustion_tick": exhausted,
            "post_exhaustion_ticks": after,
        }
        if ticks is not None:
            planned = ticks if planned is None else max(planned, int(ticks))
    exhaustions = [entry["exhaustion_tick"] for entry in conditions.values()]
    if planned is None:
        raise TrioIntegrityError(f"{directory}: no run_end event; the trio never finished")
    return {
        "planned_horizon": planned,
        "common_active_horizon": common_active_horizon(exhaustions, planned),
        "conditions": conditions,
    }


def write_integrity_marker(directory: Path | str, report: TrioIntegrityReport) -> Path:
    """Record that a trio passed the gate, pinned to the exact bytes it checked.

    Each log's SHA-256 is stored, so a log replaced or edited after the check
    no longer carries the verdict. With it goes the trio's common active
    horizon, each condition's exhaustion tick and its post-exhaustion
    mechanical ticks (``active_horizons``). Written through a temporary file
    and a rename, so a crash leaves either no marker or a whole one.
    """
    directory = Path(directory)
    payload = {
        "checks": report.to_payload(),
        "horizon": active_horizons(directory),
        "logs": {
            condition.value: _file_sha256(directory / f"{condition.value}.jsonl")
            for condition in (Condition.A, Condition.B, Condition.C)
        },
    }
    target = directory / INTEGRITY_MARKER
    partial = directory / f"{INTEGRITY_MARKER}.partial"
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(target)
    return target


def has_integrity_marker(directory: Path | str) -> bool:
    """Whether the trio passed the gate on exactly the logs it holds now."""
    directory = Path(directory)
    try:
        payload = json.loads((directory / INTEGRITY_MARKER).read_text(encoding="utf-8"))
        recorded = payload["logs"]
        for condition in (Condition.A, Condition.B, Condition.C):
            path = directory / f"{condition.value}.jsonl"
            if recorded.get(condition.value) != _file_sha256(path):
                return False
    except OSError, ValueError, KeyError, TypeError, AttributeError:
        return False
    return True


def clear_integrity_marker(directory: Path | str) -> None:
    """Drop a trio's verdict before any of its logs is replaced."""
    (Path(directory) / INTEGRITY_MARKER).unlink(missing_ok=True)
