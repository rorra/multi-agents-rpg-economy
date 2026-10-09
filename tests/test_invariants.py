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


@pytest.mark.invariant
def test_run_metadata_records_every_inference_parameter() -> None:
    """Design v1.0 s15: a run records the inference parameters that produced it.

    ``reasoning_effort`` decides whether the model answers at all -- at the
    provider default gpt-5-nano spends its whole completion budget reasoning and
    returns empty content (design s13). A log without it cannot explain its own
    run.
    """
    from rpg_economy.cli_support import build_metadata
    from tests.conftest import make_config

    params = build_metadata(make_config()).llm_params
    for key in ("temperature", "reasoning_effort", "max_output_tokens", "seed", "model_snapshot"):
        assert key in params, f"run metadata omits inference parameter {key!r}"


@pytest.mark.invariant
def test_the_runner_delegates_the_mock_decision_to_build_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run must be what its metadata says it is.

    ``run_condition`` must honour ``config.llm.mock`` through ``build_client``.
    Otherwise a ``mock: False`` configuration could run offline while its
    metadata records ``llm_mock: False``: a run that looks real and is not
    (design s12).

    Asserted through a spy rather than by running, because the real path would
    otherwise reach the network from the test suite.
    """
    import rpg_economy.runner as runner_module
    from rpg_economy.llm.client import MockLLMClient
    from rpg_economy.runner import run_condition
    from tests.conftest import make_config

    seen: list[bool] = []
    real_build = runner_module.build_client

    def spy(llm_config, responder=None):
        seen.append(llm_config.mock)
        # Always hand back a mock so the test never reaches a provider.
        return MockLLMClient(responder, model=f"mock:{llm_config.model}")

    monkeypatch.setattr(runner_module, "build_client", spy)
    run_condition(
        make_config(run={"condition": "A"}, llm={"mock": False}),
        tmp_path / "run.jsonl",
        cycles=1,
    )

    assert seen, "run_condition never consulted build_client"
    assert all(flag is False for flag in seen), (
        "run_condition passed a mock=True config to build_client for a mock=False run"
    )
    assert real_build is not spy


# ---------------------------------------------------------------------------
# The suite-wide guard against real provider calls (tests/conftest.py)
# ---------------------------------------------------------------------------
def test_no_test_can_construct_the_real_provider_client() -> None:
    """Guards the guard: ``.env`` is loaded inside the client, so the key cannot be relied on."""
    from rpg_economy.llm.client import OpenAIClient, build_client
    from tests.conftest import RealProviderCall, make_config

    config = make_config(llm={"mock": False}).llm
    with pytest.raises(RealProviderCall):
        OpenAIClient(config)
    with pytest.raises(RealProviderCall):
        build_client(config)


def test_the_provider_sdk_is_blocked_too() -> None:
    """A client built some other way still cannot be constructed or send."""
    openai = pytest.importorskip("openai")
    from openai import _base_client

    from tests.conftest import RealProviderCall

    with pytest.raises(RealProviderCall):
        openai.OpenAI(api_key="not-a-key")
    with pytest.raises(RealProviderCall):
        _base_client.SyncAPIClient.request(object(), None, None)


def test_credentials_are_never_loaded_in_tests() -> None:
    dotenv = pytest.importorskip("dotenv")
    assert dotenv.load_dotenv() is False


@pytest.mark.provider_client
def test_the_marker_admits_the_client_around_a_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Marked, OpenAIClient can be constructed -- but only around a fake SDK."""
    from types import SimpleNamespace

    openai = pytest.importorskip("openai")
    from rpg_economy.llm.client import OpenAIClient
    from tests.conftest import RealProviderCall, make_config

    config = make_config(llm={"mock": False}).llm
    with pytest.raises(RealProviderCall):
        OpenAIClient(config)  # the real SDK is still blocked

    fake = SimpleNamespace(chat=SimpleNamespace(completions=None))
    monkeypatch.setattr(openai, "OpenAI", lambda **options: fake)
    assert OpenAIClient(config)._client is fake


def test_the_marker_does_not_outlive_its_test() -> None:
    """Runs after the marked test above: the client is blocked again."""
    from rpg_economy.llm.client import OpenAIClient
    from tests.conftest import RealProviderCall, make_config

    with pytest.raises(RealProviderCall):
        OpenAIClient(make_config(llm={"mock": False}).llm)


@pytest.fixture(scope="module")
def _client_built_in_a_module_fixture() -> BaseException | None:
    """What constructing the real client in a module-scoped fixture does."""
    from rpg_economy.llm.client import OpenAIClient
    from tests.conftest import make_config

    try:
        OpenAIClient(make_config(llm={"mock": False}).llm)
    except Exception as exc:  # the test inspects it
        return exc
    return None


def test_the_guard_covers_module_scoped_fixtures(
    _client_built_in_a_module_fixture: BaseException | None,
) -> None:
    from tests.conftest import RealProviderCall

    assert isinstance(_client_built_in_a_module_fixture, RealProviderCall)


# ---------------------------------------------------------------------------
# The config hash: what a run is, not how it is analysed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field, value",
    [
        ("recovery_band_pct_x", "0.33"),
        ("recovery_hold_ticks_m", 7),
        ("useful_production_ticks_t", 9),
        ("unanswered_request_ticks_d", 11),
    ],
)
def test_analysis_only_metric_parameters_stay_out_of_the_hash(field: str, value: object) -> None:
    """Re-tuning them re-analyses finished runs; it must never re-run them."""
    from rpg_economy.config import config_hash, setup_hash
    from tests.conftest import make_config

    base = make_config()
    retuned = make_config(metrics={field: value})
    assert getattr(retuned.metrics, field) != getattr(base.metrics, field)
    assert config_hash(retuned) == config_hash(base)
    assert setup_hash(retuned) == setup_hash(base)


def test_the_volatility_window_stays_in_the_hash() -> None:
    """The global view reads W while the run executes, so it is part of what a run is."""
    from rpg_economy.config import config_hash, setup_hash
    from rpg_economy.config.loader import ANALYSIS_ONLY_METRIC_FIELDS
    from tests.conftest import make_config

    base = make_config()
    assert "volatility_window_w" not in ANALYSIS_ONLY_METRIC_FIELDS
    wider = make_config(metrics={"volatility_window_w": base.metrics.volatility_window_w + 1})
    assert config_hash(wider) != config_hash(base)
    assert setup_hash(wider) != setup_hash(base)


def test_the_analysis_only_metric_fields_exist() -> None:
    """A renamed field would silently fall back into the hash."""
    from rpg_economy.config.loader import ANALYSIS_ONLY_METRIC_FIELDS
    from rpg_economy.config.models import MetricsConfig

    assert set(ANALYSIS_ONLY_METRIC_FIELDS) <= set(MetricsConfig.model_fields)
