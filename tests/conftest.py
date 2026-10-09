"""Shared fixtures, and the suite-wide guard against real provider calls."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.config import config_hash, load_config
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Role
from rpg_economy.engine import AgentAccount, Engine, Market, WorldState, build_world
from rpg_economy.logging_ import (
    Environment,
    EventLogWriter,
    EventType,
    RunMetadata,
    read_events,
)

#: The repository root, so no test depends on the directory pytest runs from.
ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "base.yaml"

#: Marks a test that constructs ``OpenAIClient`` on purpose, around a fake SDK
#: it installs itself. The SDK and its transport stay blocked all the same.
PROVIDER_CLIENT_MARKER = "provider_client"


class RealProviderCall(AssertionError):
    """A test reached the real provider client, or the SDK behind it."""


def _refuse_provider(*args: object, **kwargs: object) -> None:
    raise RealProviderCall(
        "a test reached the real LLM provider; tests run offline with the mock client "
        f"(mark the test '{PROVIDER_CLIENT_MARKER}' to construct OpenAIClient around a fake SDK)"
    )


#: The suite-wide provider guard, installed once before collection and undone
#: after the session (``pytest_configure`` / ``pytest_unconfigure``).
_provider_guard = pytest.MonkeyPatch()

#: ``OpenAIClient.__init__`` as defined, so a ``provider_client`` test can have
#: it back for the length of that one test.
_real_client_init: object = None


def pytest_configure(config: pytest.Config) -> None:
    """No code under test can reach the provider, whatever the environment holds.

    Installed once for the whole session, before collection, so module- and
    session-scoped fixtures and import-time code are covered too, not only the
    body of each test. ``OpenAIClient.__init__`` loads credentials from ``.env``
    itself, so an unset ``OPENAI_API_KEY`` protects nothing. Blocked here:
    constructing the client; reading ``.env``; constructing the SDK's clients;
    and the SDK's request path, so even a client built some other way cannot
    send. A test marked ``provider_client`` gets ``OpenAIClient.__init__`` back
    (``_provider_client_marker``) -- it has to install a fake ``openai.OpenAI``
    to get past the block below. Tests that assemble a client around a fake
    transport without ``__init__`` need no marker.
    """
    global _real_client_init
    import rpg_economy.llm.client as client_module

    _real_client_init = client_module.OpenAIClient.__init__
    _provider_guard.setattr(client_module.OpenAIClient, "__init__", _refuse_provider)

    try:
        import dotenv
    except ImportError:  # pragma: no cover -- offline installs
        pass
    else:
        _provider_guard.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)

    try:
        import openai
        from openai import _base_client
    except ImportError:  # pragma: no cover -- offline installs
        return
    for name in ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"):
        if hasattr(openai, name):
            _provider_guard.setattr(openai, name, _refuse_provider)
    for name in ("SyncAPIClient", "AsyncAPIClient"):
        base = getattr(_base_client, name, None)
        if base is not None and hasattr(base, "request"):
            _provider_guard.setattr(base, "request", _refuse_provider)


def pytest_unconfigure(config: pytest.Config) -> None:
    _provider_guard.undo()


@pytest.fixture(autouse=True)
def _provider_client_marker(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``provider_client`` test may construct ``OpenAIClient``; only that test."""
    if request.node.get_closest_marker(PROVIDER_CLIENT_MARKER) is None:
        return
    import rpg_economy.llm.client as client_module

    monkeypatch.setattr(client_module.OpenAIClient, "__init__", _real_client_init)


@pytest.fixture(autouse=True)
def _default_locale(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test starts in the default locale, whatever the shell exports.

    ``SIM_LOCALE`` would otherwise pick the catalog for any test that does not
    pass ``--locale``, and a locale set by one test (``set_locale``) would
    carry into the next through the cached catalogs.
    """
    from rpg_economy.i18n import clear_cache

    monkeypatch.delenv("SIM_LOCALE", raising=False)
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _absolute_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI's default --config is relative to where it runs; the tests are not."""
    import rpg_economy.cli

    monkeypatch.setattr(rpg_economy.cli, "DEFAULT_CONFIG", BASE_CONFIG)


def make_config(**overrides: object) -> SimulationConfig:
    return load_config(BASE_CONFIG, overrides=dict(overrides))


def gold_flow(log_path: Path) -> Decimal:
    """Net gold entering the economy over a run: loot and confiscation pay less taxes and storage.

    The only places gold is created or destroyed. A conservation check
    reconciles against this instead of expecting a constant total.
    """
    flow = Decimal(0)
    for event in read_events(log_path):
        if event.type is EventType.EXPEDITION_RESOLVED:
            flow += Decimal(event.payload.get("gold", "0"))
        elif event.type is EventType.GOODS_CONFISCATED:
            flow += Decimal(event.payload["gold"])
        elif event.type in (EventType.TAX_PAID, EventType.STORAGE_PAID):
            flow -= Decimal(event.payload["amount"])
    return flow


def make_metadata(config: SimulationConfig) -> RunMetadata:
    digest = config_hash(config)
    return RunMetadata(
        run_id=RunMetadata.make_run_id(
            config_hash=digest,
            condition=config.run.condition,
            scenario=config.run.scenario,
            experiment=config.run.experiment,
            seed=config.run.seed,
        ),
        design_version=config.design_version,
        schema_version=config.schema_version,
        condition=config.run.condition,
        scenario=config.run.scenario,
        experiment=config.run.experiment,
        seed=config.run.seed,
        config_hash=digest,
        llm_provider=config.llm.provider,
        llm_model=config.llm.model,
        llm_params={"temperature": str(config.llm.temperature)},
        llm_mock=config.llm.mock,
        environment=Environment.capture(),
    )


@pytest.fixture
def config() -> SimulationConfig:
    return make_config()


@pytest.fixture
def two_traders() -> WorldState:
    """A minimal world: one buyer and one seller, each solvent and stocked."""
    state = WorldState()
    for agent_id in ("woodworking_1", "woodworking_2"):
        state.accounts[agent_id] = AgentAccount(
            agent_id=agent_id,
            role=Role.ARTISAN,
            specialization="woodworking",
            gold=Decimal("1000"),
            inventory={},
        )
    return state


@pytest.fixture
def market(two_traders: WorldState) -> Market:
    return Market(two_traders, order_expiry_ticks=10, visible_last_n_trades=10)


@pytest.fixture
def engine(tmp_path: Path, config: SimulationConfig) -> Iterator[Engine]:
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        yield Engine(config, state, log)
