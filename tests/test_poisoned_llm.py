"""The poisoned-mock gate: no LLM is reachable from triggers or scheduling.

Design v1.0 s2 confines semantic decisions to identified LLM calls; the scheduler,
the triggers and order matching apply fixed rules with no interpretation
(s8.2, s8.3). IMPLEMENTATION.md section 4.5 requires this be proven with a mock that
raises if it is ever called during trigger evaluation or scheduling.

The point is not that today's code happens not to call a model. It is that a
future change which introduces such a call fails loudly here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import pytest

from rpg_economy.engine import Engine, build_world
from rpg_economy.logging_ import EventLogWriter, EventType, read_events
from rpg_economy.orchestration import DecentralizedOrchestrator, NeutralScheduler
from tests.conftest import ROOT, make_config, make_metadata


class LLMWasCalled(AssertionError):
    """Raised if anything reaches the language model where it must not."""


def _detonate(*args: Any, **kwargs: Any) -> NoReturn:
    raise LLMWasCalled(
        "an LLM client was built or called during mechanical activation logic; "
        "triggers and scheduling must never consult a model (design v1.0 s8.2, s8.3)"
    )


@pytest.fixture
def poisoned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every route to a model detonates: building a client, the mock's call, the provider."""
    import rpg_economy.llm
    import rpg_economy.llm.client
    import rpg_economy.runner

    monkeypatch.setattr(rpg_economy.llm.client, "build_client", _detonate)
    monkeypatch.setattr(rpg_economy.llm, "build_client", _detonate)
    monkeypatch.setattr(rpg_economy.runner, "build_client", _detonate)
    monkeypatch.setattr(rpg_economy.llm.client.MockLLMClient, "complete", _detonate)
    monkeypatch.setattr(rpg_economy.llm.client.OpenAIClient, "__init__", _detonate)


def test_the_patched_routes_actually_detonate(poisoned) -> None:
    """Guards the guard: a poison that never fires would prove nothing."""
    from rpg_economy.llm import always
    from rpg_economy.llm.client import MockLLMClient, OpenAIClient, build_client

    with pytest.raises(LLMWasCalled):
        MockLLMClient(always("x")).complete(system="s", user="u")
    with pytest.raises(LLMWasCalled):
        OpenAIClient(make_config().llm)
    with pytest.raises(LLMWasCalled):
        build_client(make_config().llm, always("x"))


@pytest.mark.invariant
@pytest.mark.parametrize("condition", ["B", "C"])
def test_real_decentralized_cycles_reach_no_llm(poisoned, tmp_path: Path, condition: str) -> None:
    """Trigger evaluation and scheduling, through the orchestrator, for real cycles.

    The activation handler does nothing, so anything that reached a model
    during those cycles came from the mechanical path itself -- and would have
    raised. C runs the runner's own decision phase, which publishes the bulletin
    as it opens, so the bulletin path a real run takes is the one under test.
    """
    from rpg_economy.runner import _with_bulletin

    config = make_config(run={"condition": condition})
    activated: list[tuple[int, str]] = []

    def no_op(engine: Engine, agent_id: str) -> None:
        activated.append((engine.tick, agent_id))

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        orchestrator = DecentralizedOrchestrator(config, engine, no_op)
        decision_phase = _with_bulletin(config, orchestrator) if condition == "C" else orchestrator

        for _ in range(12):
            engine.run_cycle(decision_phase)

    events = list(read_events(log.path))
    kinds = {e.type for e in events}
    assert EventType.TRIGGER_FIRED in kinds, "no trigger fired; the test proves nothing"
    assert EventType.ACTIVATION_SERVED in kinds, "nothing was scheduled; the test proves nothing"
    assert activated, "the scheduler served nobody; the test proves nothing"
    assert (EventType.BULLETIN_PUBLISHED in kinds) is (condition == "C")


#: The mechanical activation path. Design v1.0 s2 confines semantic decisions to
#: identified LLM calls and requires the scheduler, the triggers and order
#: matching to apply fixed rules -- so these modules must not so much as import a
#: client. The Director is deliberately excluded: it *is* an identified LLM, and
#: Condition A's whole design is that a model chooses who acts.
MECHANICAL_MODULES = (
    "src/rpg_economy/orchestration/triggers.py",
    "src/rpg_economy/orchestration/scheduler.py",
    "src/rpg_economy/orchestration/activation.py",
    "src/rpg_economy/orchestration/decentralized.py",
    "src/rpg_economy/orchestration/global_view.py",
    "src/rpg_economy/engine/market.py",
)


@pytest.mark.invariant
def test_the_mechanical_path_imports_no_llm_module() -> None:
    """Structural companion to the runtime check."""
    import ast

    for name in MECHANICAL_MODULES:
        path = ROOT / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for imported in names:
                # The prohibition is on *reaching a model*, which means the
                # client. Token accounting and the budget are experimental
                # bookkeeping: asking whether a call is affordable is not
                # consulting one, and the orchestrator must be able to stop
                # serving activations when the budget is gone (design s8.4).
                assert not imported.startswith("rpg_economy.llm.client"), (
                    f"{name} imports {imported}; triggers, scheduling and matching "
                    f"must never consult a model (design v1.0 s2, s8.2, s8.3)"
                )


@pytest.mark.invariant
def test_the_director_is_the_only_orchestration_module_using_a_model() -> None:
    """Condition A's coordinator is an identified LLM; nothing else there is.

    Measured by who imports the *client*. Several modules import the token
    ledger or the budget, which are accounting, not access to a model.
    """
    import ast

    users = []
    for path in sorted((ROOT / "src/rpg_economy/orchestration").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n.startswith("rpg_economy.llm.client") for n in names):
                users.append(path.name)
    assert sorted(set(users)) == ["director.py"]


@pytest.mark.invariant
def test_the_scheduler_cannot_inspect_state_or_messages() -> None:
    """Neutrality is structural: ``serve`` takes ids and nothing else.

    Design v1.0 s8.3 forbids the scheduler from inspecting state or message
    content. Asserting it by signature is stronger than asserting it by
    behaviour: there is no parameter through which content could arrive.
    """
    import inspect

    signature = inspect.signature(NeutralScheduler.serve)
    assert list(signature.parameters) == ["self", "requesting"]

    source = inspect.getsource(NeutralScheduler)
    for forbidden in ("WorldState", "AgentAccount", "Message", "payload", "inventory", "gold"):
        assert forbidden not in source, f"the scheduler references {forbidden!r}"
