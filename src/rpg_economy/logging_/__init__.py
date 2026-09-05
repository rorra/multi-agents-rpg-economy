"""Append-only JSONL event logging and run metadata."""

from rpg_economy.logging_.events import Event, EventType, prompt_sha
from rpg_economy.logging_.metadata import (
    SUPPORTED_PYTHON_SERIES,
    Environment,
    RunMetadata,
    code_digest,
    current_python_version,
)
from rpg_economy.logging_.writer import (
    EventLogWriter,
    LogError,
    last_event,
    log_digest,
    log_is_finished,
    read_events,
    read_metadata,
    run_logs,
)

__all__ = [
    "SUPPORTED_PYTHON_SERIES",
    "Environment",
    "Event",
    "EventLogWriter",
    "EventType",
    "LogError",
    "RunMetadata",
    "code_digest",
    "current_python_version",
    "last_event",
    "log_digest",
    "log_is_finished",
    "prompt_sha",
    "read_events",
    "read_metadata",
    "run_logs",
]
