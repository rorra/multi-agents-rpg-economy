"""Structural invariants that must hold from phase 0 onward.

These guard properties that are cheap to preserve and expensive to retrofit once
the engine exists.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Anchored to this file, not the working directory, so the suite checks the
#: same tree wherever pytest is launched from.
REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "rpg_economy"

#: Modules that reach the network or read credentials.
NETWORK_MODULES = frozenset(
    {
        "openai",
        "anthropic",
        "requests",
        "httpx",
        "urllib",
        "urllib3",
        "http",
        "socket",
        "aiohttp",
        "dotenv",
    }
)

#: The only package permitted to import them (IMPLEMENTATION.md section 3).
NETWORK_PERMITTED_PACKAGE = "llm"


def _imported_roots(path: Path) -> set[str]:
    """Return the top-level module names imported by ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _package_of(path: Path) -> str:
    relative = path.relative_to(SRC)
    return relative.parts[0] if len(relative.parts) > 1 else ""


def test_only_the_llm_package_touches_the_network() -> None:
    """The engine must run fully offline; nothing else may import a client."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if _package_of(path) == NETWORK_PERMITTED_PACKAGE:
            continue
        for module in sorted(_imported_roots(path) & NETWORK_MODULES):
            violations.append(f"{path}: imports {module}")
    assert not violations, "network imports outside rpg_economy.llm:\n" + "\n".join(violations)


def test_the_engine_does_not_depend_on_the_llm_package() -> None:
    """LLMs propose; the engine validates and executes. The dependency is one-way."""
    violations: list[str] = []
    for path in sorted((SRC / "engine").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("rpg_economy.llm") or name.startswith("rpg_economy.agents"):
                    violations.append(f"{path}: imports {name}")
    assert not violations, "engine reaches outside its layer:\n" + "\n".join(violations)


def test_gitignore_excludes_environment_files() -> None:
    """Credentials must never become committable (IMPLEMENTATION.md section 3)."""
    patterns = {
        line.strip()
        for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert ".env" in patterns
    assert ".env.*" in patterns


def test_no_env_file_is_tracked_by_git() -> None:
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=False, cwd=REPO
    ).stdout.splitlines()
    offenders = [
        f for f in tracked if Path(f).name.startswith(".env") and Path(f).name != ".env.example"
    ]
    assert not offenders, f"environment files are tracked: {offenders}"


def test_design_document_is_present_and_unmodified_in_name() -> None:
    assert (REPO / "docs/design/DESIGN_v1.0.md").is_file()


def test_running_interpreter_is_the_supported_series() -> None:
    """The project targets a single interpreter series (IMPLEMENTATION.md section 3)."""
    import sys

    from rpg_economy.logging_ import SUPPORTED_PYTHON_SERIES

    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert running == SUPPORTED_PYTHON_SERIES, (
        f"running on Python {running}; this project supports only {SUPPORTED_PYTHON_SERIES}.x"
    )


def test_packaging_metadata_pins_the_supported_series() -> None:
    """pyproject, CI and the code must agree on the supported interpreter."""
    import tomllib

    import yaml

    from rpg_economy.logging_ import SUPPORTED_PYTHON_SERIES

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    major, minor = (int(part) for part in SUPPORTED_PYTHON_SERIES.split("."))
    # The series and nothing after it: a bare lower bound would admit 3.15.
    assert pyproject["project"]["requires-python"] == (
        f">={SUPPORTED_PYTHON_SERIES},<{major}.{minor + 1}"
    )

    series_tag = "py" + SUPPORTED_PYTHON_SERIES.replace(".", "")
    assert pyproject["tool"]["ruff"]["target-version"] == series_tag

    # Every interpreter any CI job sets up, read from the workflow structure
    # rather than searched for as text: exactly one, and it is the series.
    versions: list[str] = []
    for workflow in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        document = yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}
        for job in (document.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                if str(step.get("uses", "")).startswith("actions/setup-python"):
                    configured = (step.get("with") or {}).get("python-version")
                    listed = configured if isinstance(configured, list) else [configured]
                    versions.extend(str(v) for v in listed)
            matrix = (job.get("strategy") or {}).get("matrix") or {}
            versions.extend(str(v) for v in matrix.get("python-version") or [])
    assert versions, "no CI job sets up Python"
    assert set(versions) == {SUPPORTED_PYTHON_SERIES}, (
        f"CI sets up {sorted(set(versions))}; only {SUPPORTED_PYTHON_SERIES} is supported"
    )


# ---------------------------------------------------------------------------
# Section 15 logging coverage (IMPLEMENTATION.md 4.12, design s15)
# ---------------------------------------------------------------------------
#: Every event type section 15 requires a run to emit, by condition. Declaring
#: an event type in the enum does not prove anything emits it.
REQUIRED_EVENTS = {
    "all": (
        "run_metadata",
        "observation_delivered",
        "agent_output",
        "action_executed",
        "llm_call",
        "tokens_consumed",
        "order_placed",
        "cycle_start",
        "cycle_end",
        "tick",
        "shock_applied",
        "run_end",
    ),
    "A": ("director_call", "director_task_assigned", "result_returned"),
    "B": ("trigger_fired", "activation_requested", "activation_served"),
    "C": ("trigger_fired", "activation_served", "bulletin_published"),
}


def _event_types(path: Path) -> set[str]:
    import json

    return {json.loads(line)["type"] for line in path.read_text().splitlines()}


@pytest.mark.invariant
def test_every_run_logs_what_section_15_requires(tmp_path: Path) -> None:
    """A run must be reconstructable from its log, so the log must contain it."""
    from rpg_economy.runner import run_trio
    from tests.conftest import make_config

    config = make_config(
        phases={"base_ticks": 6, "shock_ticks": 6, "recovery_ticks": 8},
        time={"horizon_ticks": 20},
        metrics={"volatility_window_w": 3, "recovery_hold_ticks_m": 2},
    )
    trio = run_trio(config, tmp_path, seed=11)

    missing: list[str] = []
    for condition, result in trio.items():
        present = _event_types(result.log_path)
        required = (*REQUIRED_EVENTS["all"], *REQUIRED_EVENTS[condition.value])
        missing += [f"{condition.value}: {name}" for name in required if name not in present]
    assert not missing, "section 15 events never logged:\n" + "\n".join(missing)


@pytest.mark.invariant
def test_the_delivered_observation_is_logged_in_full_not_hashed(tmp_path: Path) -> None:
    """Design s15: a hash verifies a replay you already have; it cannot produce one.

    The bulletin and inbox are logged too -- they are what distinguish the
    conditions.
    """
    import json

    from rpg_economy.domain import Condition
    from rpg_economy.runner import run_trio
    from tests.conftest import make_config

    config = make_config(
        phases={"base_ticks": 6, "shock_ticks": 6, "recovery_ticks": 8},
        time={"horizon_ticks": 20},
        metrics={"volatility_window_w": 3, "recovery_hold_ticks_m": 2},
    )
    trio = run_trio(config, tmp_path, seed=11)

    required = {
        "agent_id",
        "gold",
        "available_gold",
        "inventory",
        "equipment",
        "market",
        "directory",
        "open_orders",
        "messages",
        "bulletin",
        "traits",
    }

    seen_bulletin = {Condition.A: False, Condition.B: False, Condition.C: False}
    for condition, result in trio.items():
        events = [
            json.loads(line)
            for line in result.log_path.read_text().splitlines()
            if '"observation_delivered"' in line
        ]
        assert events, f"{condition.value} logged no delivered observation"
        for event in events:
            observation = event["payload"]["observation"]
            assert required <= set(observation)
            if observation["bulletin"] is not None:
                seen_bulletin[condition] = True

    # The log must show the observability ablation, not merely claim it.
    assert seen_bulletin[Condition.C]
    assert not seen_bulletin[Condition.B]
    assert not seen_bulletin[Condition.A]
