"""Run metadata -- the reproducibility record required by design v1.0 s15.

Everything needed to re-run an experiment identically is recorded here and
written as the first line of the run log, so that a single file is sufficient to
reconstruct the run.

The ``environment`` block is deliberately segregated: it holds the only
non-deterministic information in the log (wall clock, host). The determinism
digest excludes it, so two runs of the same seed still compare byte for byte
while the provenance information remains available.

``python_version`` is deliberately *not* in that block. The project targets a
single interpreter series, which makes the exact ``3.14.x`` patch release part of
the experiment's environment rather than incidental provenance: a rerun
under a different patch is not the same environment, and the digest should say
so (IMPLEMENTATION.md section 3).
"""

from __future__ import annotations

import functools
import hashlib
import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rpg_economy.config.hashing import canonical_hash
from rpg_economy.domain import Condition, ExperimentRegime, Scenario

#: Metadata keys excluded from the determinism digest.
NON_DETERMINISTIC_KEYS = frozenset({"environment"})

#: The single supported interpreter series (IMPLEMENTATION.md section 3).
SUPPORTED_PYTHON_SERIES = "3.14"


def current_python_version() -> str:
    """Return the running interpreter's exact ``major.minor.patch`` version."""
    info = sys.version_info
    return f"{info.major}.{info.minor}.{info.micro}"


#: The parts of the package that determine what a run does: packages as
#: directories, single modules as files, relative to the package root. Analysis,
#: metrics, the dashboard, i18n, the CLI, campaign orchestration and the trio
#: integrity check read or drive runs but do not change what a run does, so an
#: edit to them does not make a run's log stale. ``cli_support`` builds the run
#: metadata; ``__init__`` is the package itself.
SIMULATION_SOURCES: tuple[str, ...] = (
    "__init__.py",
    "agents",
    "cli_support.py",
    "config",
    "domain.py",
    "engine",
    "llm",
    "logging_",
    "messaging",
    "orchestration",
    "runner.py",
)


def _simulation_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in SIMULATION_SOURCES:
        entry = root / name
        if entry.is_dir():
            files.extend(entry.rglob("*.py"))
        elif entry.is_file():
            files.append(entry)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


@functools.cache
def code_digest() -> str:
    """Return a sha256 over the simulation code's Python source.

    Only the modules that determine run behaviour contribute
    (``SIMULATION_SOURCES``): each ``.py`` file contributes its path relative
    to the package directory and its bytes, in sorted path order, so the digest
    changes when any such module is added, removed, renamed or edited, and is
    independent of where the package is installed. It identifies the code that
    produced a run: resuming a run or deriving a budget from logs written by
    different code is then detectable rather than silent.

    Computed once per process and cached: the code a process runs is the code it
    imported, so a file edited on disk mid-campaign must not be recorded as the
    running code. Tests that edit sources call ``code_digest.cache_clear()``.
    """
    import rpg_economy

    root = Path(rpg_economy.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in _simulation_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        # Length-prefix both fields so no path/content split is ambiguous.
        for part in (relative, content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


#: The clear functions of the per-process prompt-hash caches, cleared together.
_HASH_CACHES: list[Callable[[], None]] = []


def register_hash_cache(clear: Callable[[], None]) -> None:
    """Register a prompt-hash cache's clear function with the others.

    The agents' and the Director's prompt hashes are each cached once per
    process, and a test that alters a template must be able to drop both: a
    cleared agent cache beside a stale Director one would record hashes of two
    different wordings in the same metadata. Registering here, a module both
    import, couples the caches without the modules importing each other.
    """
    if clear not in _HASH_CACHES:
        _HASH_CACHES.append(clear)


def clear_hash_caches() -> None:
    """Clear every registered prompt-hash cache."""
    for clear in _HASH_CACHES:
        clear()


class Environment(BaseModel):
    """Provenance, excluded from the determinism digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    started_at: str
    platform: str

    @classmethod
    def capture(cls) -> Environment:
        return cls(
            started_at=datetime.now(UTC).isoformat(),
            platform=platform.platform(),
        )


class RunMetadata(BaseModel):
    """The run's identity and full parameter provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(description="Deterministic identity; equal for identical reruns.")
    design_version: str
    schema_version: str

    condition: Condition
    scenario: Scenario
    experiment: ExperimentRegime
    seed: int
    phase: str | None = Field(
        default=None,
        description="The campaign phase the run belongs to (config run.phase: "
        "'pilot', 'final' or 'adhoc'); None in logs written before it was recorded.",
    )

    config_hash: str
    setup_hash: str | None = Field(
        default=None,
        description="Hash of the configuration without its run section: the "
        "setup this run derives from.",
    )
    python_version: str = Field(
        default_factory=current_python_version,
        description="Exact interpreter version, e.g. '3.14.7'. Part of the "
        "environment and therefore of the determinism digest, not of `environment`.",
    )
    code_digest: str | None = Field(
        default=None,
        description="sha256 of the simulation code's Python source "
        "(logging_.metadata.code_digest); None in logs written before it was recorded.",
    )
    prompt_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="Version hash per prompt template (design v1.0 s15).",
    )
    global_view_version: str | None = Field(
        default=None,
        description="Format version of the global view the Director reads in A and "
        "the bulletin carries in C (design v1.0 s5.2, s3.3).",
    )

    llm_provider: str
    llm_model: str
    llm_params: dict[str, Any] = Field(default_factory=dict)
    llm_mock: bool

    environment: Environment

    @staticmethod
    def make_run_id(
        *,
        config_hash: str,
        condition: Condition,
        scenario: Scenario,
        experiment: ExperimentRegime,
        seed: int,
    ) -> str:
        """Derive a run id from the run's identity.

        Deterministic on purpose: rerunning the same configuration and seed must
        produce the same id, or the byte-identical-log test could never pass. It
        is still unique across distinct runs, since it hashes the full identity.
        """
        return canonical_hash(
            {
                "config_hash": config_hash,
                "condition": str(condition),
                "scenario": str(scenario),
                "experiment": str(experiment),
                "seed": seed,
            }
        )[:16]
