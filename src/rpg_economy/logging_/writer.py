"""Append-only JSONL event log writer and reader.

Analysis reads logs; it never mutates them (IMPLEMENTATION.md section 3). The writer
enforces that on its side: it opens for append only, never truncates, refuses to
attach to a log that already holds events, and assigns gapless sequence numbers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from rpg_economy.config.hashing import canonical_hash, canonical_json
from rpg_economy.logging_.events import Event, EventType
from rpg_economy.logging_.metadata import NON_DETERMINISTIC_KEYS, RunMetadata


class LogError(Exception):
    """Raised on any attempt to use a run log in a way that would corrupt it."""


class EventLogWriter:
    """Writes a run's events as newline-delimited JSON.

    Use as a context manager. The run metadata is written as the first event, so
    that the log is self-describing and a run is reconstructable from it alone.
    """

    def __init__(self, path: Path | str, metadata: RunMetadata) -> None:
        self._path = Path(path)
        self._metadata = metadata
        self._seq = 0
        self._handle: Any = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> Self:
        if self._path.exists() and self._path.stat().st_size > 0:
            raise LogError(
                f"refusing to append to non-empty run log {self._path}: run logs are "
                f"write-once. Choose a new path."
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered: every event reaches the file as soon as it is written,
        # so a run that is killed or crashes still leaves the record of every
        # call it made, including the ones already billed.
        self._handle = self._path.open("a", encoding="utf-8", buffering=1)
        self.write(
            EventType.RUN_METADATA,
            tick=0,
            cycle=0,
            payload=self._metadata.model_dump(mode="json"),
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None and not self._closed:
            self._handle.flush()
            self._handle.close()
            self._closed = True

    # -- writing -----------------------------------------------------------
    def write(
        self,
        event_type: EventType,
        *,
        tick: int,
        cycle: int,
        agent_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        """Append one event and return it."""
        if self._handle is None or self._closed:
            raise LogError("event log is not open")
        event = Event(
            seq=self._seq,
            tick=tick,
            cycle=cycle,
            type=event_type,
            agent_id=agent_id,
            payload=payload or {},
        )
        self._handle.write(canonical_json(event.model_dump(mode="json")) + "\n")
        self._seq += 1
        return event

    @property
    def path(self) -> Path:
        return self._path

    @property
    def events_written(self) -> int:
        return self._seq


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_events(path: Path | str) -> Iterator[Event]:
    """Yield every event of a run log in order, validating each line."""
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Event.model_validate(json.loads(line))
            except Exception as exc:
                raise LogError(f"{path}:{lineno}: malformed event: {exc}") from exc


def read_metadata(path: Path | str) -> RunMetadata:
    """Return the run metadata recorded as the log's first event."""
    for event in read_events(path):
        if event.type is EventType.RUN_METADATA:
            return RunMetadata.model_validate(event.payload)
        break
    raise LogError(f"{path}: first event is not {EventType.RUN_METADATA}")


def log_digest(path: Path | str) -> str:
    """Return a determinism digest of a run log.

    Non-deterministic provenance (the metadata ``environment`` block) is
    excluded, so two runs of the same seed and configuration must produce equal
    digests. This is the value the determinism test compares.
    """
    records: list[dict[str, Any]] = []
    for event in read_events(path):
        record = event.model_dump(mode="json")
        if event.type is EventType.RUN_METADATA:
            record["payload"] = {
                key: value
                for key, value in record["payload"].items()
                if key not in NON_DETERMINISTIC_KEYS
            }
        records.append(record)
    return canonical_hash(records)


#: How much of a log's end to read when checking that it finished.
_TAIL_BYTES = 64 * 1024


def last_event(path: Path | str) -> dict[str, Any] | None:
    """The final complete event of a log, read from its tail; None if there is none.

    A line cut off mid-write is not an event, so a log whose last byte is not a
    newline has no trustworthy final event.
    """
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size == 0:
            return None
        handle.seek(max(0, size - _TAIL_BYTES))
        tail = handle.read()
    if not tail.endswith(b"\n"):
        return None
    try:
        event = json.loads(tail.rstrip(b"\n").rsplit(b"\n", 1)[-1])
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def log_is_finished(path: Path | str) -> bool:
    """Whether a run log ends cleanly with its ``run_end`` event.

    A run that crashed, was killed or lost its provider leaves a log that stops
    mid-way, possibly mid-line. Existence or size says nothing about that; only
    the closing event does (IMPLEMENTATION.md section 5).
    """
    event = last_event(path)
    return event is not None and event.get("type") == EventType.RUN_END.value


#: Directory prefix for runs set aside before a re-run; never read as results.
SET_ASIDE_PREFIX = "previous-"


def run_logs(directory: Path | str) -> list[Path]:
    """Every run log under ``directory``, leaving out runs set aside.

    A trio that was re-run keeps its earlier logs in ``previous-N/``; they are
    evidence of what was spent, not results, and must never be counted twice.
    """
    root = Path(directory)
    return sorted(
        path
        for path in root.rglob("*.jsonl")
        if not any(part.startswith(SET_ASIDE_PREFIX) for part in path.relative_to(root).parts)
    )
