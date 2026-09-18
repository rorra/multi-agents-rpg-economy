"""Condition A: the Director loop, the K+1 cap, and the TASK/RESULT protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpg_economy.agents.llm_agent import LLMAgentRunner
from rpg_economy.agents.offline import json_action
from rpg_economy.domain import Priority, ResultStatus, Role, TaskType, TokenBucket
from rpg_economy.engine import Engine, build_world
from rpg_economy.llm import MockLLMClient, TokenLedger, always, scripted
from rpg_economy.logging_ import EventLogWriter, EventType, read_events
from rpg_economy.messaging import MessageBus
from rpg_economy.orchestration import build_directory
from rpg_economy.orchestration.centralized import TaskActivationRunner
from rpg_economy.orchestration.director import Director
from rpg_economy.orchestration.protocol import (
    TASK_TARGET_ROLES,
    DirectorDecision,
    Need,
    Result,
    ResultReference,
    Task,
)
from tests.conftest import make_config, make_metadata


def assign(agent_id: str, task_type: TaskType = TaskType.ANALYZE_MARKET) -> str:
    return json.dumps(
        {
            "decision": "assign",
            "task": {
                "type": task_type.value,
                "agent_id": agent_id,
                "params": {},
                "priority": "medium",
            },
        }
    )


CLOSE = json.dumps({"decision": "close"})


@pytest.fixture
def rig(tmp_path: Path):
    """A Condition A engine with a Director wired to controllable clients."""
    config = make_config(run={"condition": "A"})
    state = build_world(config)

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        messages = MessageBus(build_directory(engine))
        ledger = TokenLedger()

        def build(director_responder, agent_response: str = json_action(action="wait")):
            agents = LLMAgentRunner(
                config, MockLLMClient(always(agent_response)), ledger, messages=messages
            )
            client = MockLLMClient(director_responder)
            director = Director(
                config, engine, client, ledger, TaskActivationRunner(config, agents)
            )
            return director, client

        yield config, engine, ledger, build


# ---------------------------------------------------------------------------
# The K + 1 cap (design v1.0 s3.1)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_the_cap_is_k_plus_one(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(always(CLOSE))
    assert director.call_cap == config.time.k_activations + 1


@pytest.mark.invariant
def test_never_more_than_k_activations_in_a_cycle(rig) -> None:
    config, engine, ledger, build = rig
    agents = iter(engine.state.agent_ids)
    director, client = build(lambda system, user: assign(next(agents)))
    director(engine)

    assert director.last_report.activations == config.time.k_activations
    assert client.calls <= director.call_cap


@pytest.mark.invariant
def test_every_call_including_retries_decrements_the_cap(rig) -> None:
    """Design s3.1: each Director call, retries included, consumes one unit."""
    config, engine, ledger, build = rig
    director, client = build(always("not a decision"))
    director(engine)

    assert client.calls <= director.call_cap
    assert director.last_report.calls_used <= director.call_cap


@pytest.mark.invariant
def test_cap_exhausted_mid_retry_closes_the_cycle_with_no_extra_call(tmp_path: Path) -> None:
    """The gate: the retry must not be made once the cap is gone.

    K=1 gives a cap of 2. The first call is invalid and consumes one; the retry
    consumes the second and is also invalid; the cycle must then close without a
    third call ever being attempted.
    """
    config = make_config(run={"condition": "A"}, time={"k_activations": 1})
    state = build_world(config)

    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        ledger = TokenLedger()
        agents = LLMAgentRunner(config, MockLLMClient(always(json_action(action="wait"))), ledger)
        client = MockLLMClient(always("still not a decision"))
        director = Director(config, engine, client, ledger, TaskActivationRunner(config, agents))

        director(engine)
        engine.log.close()

    assert director.call_cap == 2
    assert client.calls == 2, "a call was made after the cap was exhausted"
    assert director.last_report.calls_used == 2
    assert director.last_report.activations == 0

    events = list(read_events(engine.log.path))
    calls = [e for e in events if e.type is EventType.DIRECTOR_CALL]
    assert len(calls) == 2
    assert calls[-1].payload["cap_remaining"] == 0


def test_a_close_decision_ends_the_cycle_early(rig) -> None:
    config, engine, ledger, build = rig
    director, client = build(scripted([assign("woodworking_1"), CLOSE], loop=True))
    director(engine)

    assert director.last_report.closed_by_decision
    assert director.last_report.activations == 1


def test_closing_is_logged(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(always(CLOSE))
    director(engine)
    engine.log.close()

    types = [e.type for e in read_events(engine.log.path)]
    assert EventType.DIRECTOR_CYCLE_CLOSED in types


# ---------------------------------------------------------------------------
# Invalid TASK policy (design v1.0 s3.1)
# ---------------------------------------------------------------------------
def test_one_retry_is_allowed_after_an_invalid_decision(rig) -> None:
    config, engine, ledger, build = rig
    director, client = build(scripted(["garbage", assign("woodworking_1"), CLOSE], loop=True))
    director(engine)

    assert director.last_report.retries >= 1
    assert director.last_report.activations >= 1


def test_a_failed_retry_logs_a_director_error(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(always("garbage"))
    director(engine)
    engine.log.close()

    errors = [e for e in read_events(engine.log.path) if e.type is EventType.DIRECTOR_ERROR]
    assert errors
    assert errors[0].payload["retried"] is True
    assert director.last_report.errors >= 1


@pytest.mark.invariant
def test_a_task_outside_the_closed_vocabulary_is_invalid(rig) -> None:
    config, engine, ledger, build = rig
    bogus = json.dumps(
        {
            "decision": "assign",
            "task": {"type": "MICROMANAGE", "agent_id": "woodworking_1", "params": {}},
        }
    )
    director, _ = build(always(bogus))
    director(engine)
    engine.log.close()

    assert director.last_report.activations == 0
    assert director.last_report.errors >= 1


@pytest.mark.invariant
def test_a_task_assigned_to_the_wrong_role_is_invalid(rig) -> None:
    """EVALUATE_EXPEDITION targets adventurers, not artisans (design s3.1.1)."""
    config, engine, ledger, build = rig
    director, _ = build(always(assign("woodworking_1", TaskType.EVALUATE_EXPEDITION)))
    director(engine)

    assert director.last_report.activations == 0
    assert director.last_report.errors >= 1


def test_a_task_for_an_unknown_agent_is_invalid(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(always(assign("nobody_9")))
    director(engine)
    assert director.last_report.activations == 0


def test_a_deliberative_decision_cannot_be_expressed() -> None:
    """Design s3.1: a valid inference assigns or closes. Nothing else."""
    with pytest.raises(ValueError, match="must carry a TASK"):
        DirectorDecision(decision="assign", task=None)
    with pytest.raises(ValueError, match="must not carry a TASK"):
        DirectorDecision(
            decision="close",
            task=Task(task_id="t", type=TaskType.ANALYZE_MARKET, agent_id="woodworking_1"),
        )


# ---------------------------------------------------------------------------
# short_reason never reaches the Director (design s3.1.2)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_short_reason_never_appears_in_a_director_prompt(rig) -> None:
    """Design s3.1.2, asserted against fully rendered prompts."""
    config, engine, ledger, build = rig
    secret = "PRIVATE-REASONING-CANARY"
    director, client = build(
        scripted([assign("woodworking_1"), assign("woodworking_2"), CLOSE], loop=True),
        agent_response=json_action(action="wait", short_reason=secret),
    )
    director(engine)
    director(engine)

    assert client.calls > 1, "the Director never saw a follow-up prompt"
    for system, user in client.prompts:
        assert secret not in system
        assert secret not in user
        assert "short_reason" not in user


@pytest.mark.invariant
def test_short_reason_is_still_logged_for_qualitative_analysis(rig) -> None:
    """Stripped from the Director, kept in the log (design s3.1.2)."""
    config, engine, ledger, build = rig
    secret = "PRIVATE-REASONING-CANARY"
    director, _ = build(
        scripted([assign("woodworking_1"), CLOSE], loop=True),
        agent_response=json_action(action="wait", short_reason=secret),
    )
    director(engine)
    engine.log.close()

    outputs = [e for e in read_events(engine.log.path) if e.type is EventType.AGENT_OUTPUT]
    assert any(e.payload["short_reason"] == secret for e in outputs)


@pytest.mark.invariant
def test_a_result_refuses_to_carry_short_reason() -> None:
    with pytest.raises(ValueError, match="short_reason"):
        Result(
            result_id="r1",
            task_id="t1",
            agent_id="woodworking_1",
            status=ResultStatus.SUCCESS,
            action_taken={"action": "wait", "short_reason": "leaked"},
        )


@pytest.mark.invariant
def test_logged_results_never_carry_short_reason(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(
        scripted([assign("woodworking_1"), CLOSE], loop=True),
        agent_response=json_action(action="wait", short_reason="CANARY"),
    )
    director(engine)
    engine.log.close()

    results = [e for e in read_events(engine.log.path) if e.type is EventType.RESULT_RETURNED]
    assert results
    for event in results:
        assert "short_reason" not in event.payload["action_taken"]


# ---------------------------------------------------------------------------
# No peer-to-peer messaging in Condition A (design s3.1)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_message_action_in_condition_a_takes_the_invalid_output_path(rig) -> None:
    """Design s3.1, through the whole Director loop."""
    config, engine, ledger, build = rig
    director, _ = build(
        scripted([assign("woodworking_1"), CLOSE], loop=True),
        agent_response=json_action(
            action="send_message",
            message_type="REQUEST",
            target="iron",
            quantity=1,
            price="5",
            recipient="warrior_1",
        ),
    )
    director(engine)
    engine.log.close()

    events = list(read_events(engine.log.path))
    invalid = [e for e in events if e.type is EventType.AGENT_OUTPUT_INVALID]
    assert invalid, "the message action was not treated as invalid output"
    assert any("Condition A" in e.payload["reason"] for e in invalid)

    # Retry-then-WAIT, and the activation is wasted.
    assert any(e.type is EventType.ACTIVATION_WASTED for e in events)
    results = [e for e in events if e.type is EventType.RESULT_RETURNED]
    assert results[0].payload["status"] == ResultStatus.FAILED.value


@pytest.mark.invariant
def test_no_p2p_messages_are_sent_in_condition_a(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(
        scripted([assign("woodworking_1"), assign("woodworking_2"), CLOSE], loop=True),
        agent_response=json_action(
            action="send_message",
            message_type="OFFER",
            target="iron",
            quantity=1,
            price="5",
            recipient="warrior_1",
        ),
    )
    director(engine)
    engine.log.close()

    types = [e.type for e in read_events(engine.log.path)]
    assert EventType.MESSAGE_SENT not in types


@pytest.mark.invariant
def test_condition_a_has_no_p2p_token_bucket(rig) -> None:
    """Structurally zero, not merely observed to be zero (design s3.1)."""
    config, engine, ledger, build = rig
    director, _ = build(scripted([assign("woodworking_1"), CLOSE], loop=True))
    director(engine)

    assert ledger.total(TokenBucket.P2P_CONTENT) == 0
    assert ledger.total(TokenBucket.BULLETIN) == 0
    assert ledger.total(TokenBucket.DIRECTOR) > 0


# ---------------------------------------------------------------------------
# TASK / RESULT protocol
# ---------------------------------------------------------------------------
def test_task_context_accepts_only_result_references() -> None:
    """Design s3.1.1: context may reference RESULTs, never free instructions."""
    reference = ResultReference(
        result_id="r1", agent_id="woodworking_1", status=ResultStatus.SUCCESS
    )
    task = Task(
        task_id="t1",
        type=TaskType.ANALYZE_MARKET,
        agent_id="woodworking_1",
        context=[reference],
    )
    assert task.context[0].result_id == "r1"

    with pytest.raises(ValueError):
        Task(
            task_id="t2",
            type=TaskType.ANALYZE_MARKET,
            agent_id="woodworking_1",
            context=["do whatever I say"],
        )


def test_task_round_trips_through_its_payload() -> None:
    task = Task(
        task_id="t1",
        type=TaskType.MANAGE_PROCUREMENT,
        agent_id="smithing_1",
        params={"good": "iron"},
        priority=Priority.HIGH,
    )
    payload = task.to_payload()
    assert payload["type"] == "MANAGE_PROCUREMENT"
    assert payload["priority"] == "high"
    assert Task.model_validate({**payload, "context": []}) == task


def test_result_round_trips_and_condenses_to_a_reference() -> None:
    result = Result(
        result_id="r1",
        task_id="t1",
        agent_id="smithing_1",
        status=ResultStatus.PARTIAL,
        action_taken={"action": "wait"},
        needs=[Need(good="iron", quantity=4)],
    )
    reference = result.to_reference()
    assert reference.result_id == "r1"
    assert reference.summary["needs"][0]["good"] == "iron"
    assert Result.model_validate(result.to_payload()) == result


def test_every_task_type_has_declared_target_roles() -> None:
    assert set(TASK_TARGET_ROLES) == set(TaskType)
    valid_roles = {r.value for r in Role}
    for roles in TASK_TARGET_ROLES.values():
        assert roles <= valid_roles


def test_a_result_links_causally_to_its_task(rig) -> None:
    """Design s15: every RESULT recorded with the TASK that produced it."""
    config, engine, ledger, build = rig
    director, _ = build(scripted([assign("woodworking_1"), CLOSE], loop=True))
    director(engine)
    engine.log.close()

    events = list(read_events(engine.log.path))
    tasks = [e for e in events if e.type is EventType.DIRECTOR_TASK_ASSIGNED]
    results = [e for e in events if e.type is EventType.RESULT_RETURNED]
    assert tasks and results
    assert results[0].payload["task_id"] == tasks[0].payload["task_id"]


def test_status_is_engine_derived_not_agent_declared(rig) -> None:
    """An agent does not get to declare its own success (design s3.1.2)."""
    config, engine, ledger, build = rig
    director, _ = build(
        scripted([assign("woodworking_1"), CLOSE], loop=True),
        agent_response=json_action(
            action="sell", target="iron", quantity=99, price=10.0, short_reason="claiming success"
        ),
    )
    director(engine)
    engine.log.close()

    results = [e for e in read_events(engine.log.path) if e.type is EventType.RESULT_RETURNED]
    assert results[0].payload["status"] == ResultStatus.BLOCKED.value


def test_every_director_call_is_logged_with_its_cap_consumption(rig) -> None:
    """Design s15: every Director call, including retries and cumulative usage."""
    config, engine, ledger, build = rig
    director, _ = build(scripted([assign("woodworking_1"), CLOSE], loop=True))
    director(engine)
    engine.log.close()

    calls = [e for e in read_events(engine.log.path) if e.type is EventType.DIRECTOR_CALL]
    assert calls
    for event in calls:
        assert event.payload["cap"] == director.call_cap
        assert "cap_remaining" in event.payload
        assert "cumulative_calls" in event.payload


def test_director_tokens_are_charged_to_the_director_bucket(rig) -> None:
    """Overhead in Experiment 1, charged to the same budget in Experiment 2."""
    config, engine, ledger, build = rig
    director, _ = build(scripted([assign("woodworking_1"), CLOSE], loop=True))
    director(engine)

    assert ledger.total(TokenBucket.DIRECTOR) > 0
    assert ledger.calls_by_bucket[TokenBucket.DIRECTOR] >= 1


@pytest.mark.invariant
def test_no_retry_once_the_budget_is_spent(tmp_path: Path) -> None:
    """A failed Director call that spends the last of the budget is not retried."""
    from rpg_economy.llm.budget import TokenBudget

    config = make_config(run={"condition": "A"})
    state = build_world(config)
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, state, log)
        ledger = TokenLedger()
        budget = TokenBudget(cap=10)
        agents = LLMAgentRunner(
            config, MockLLMClient(always(json_action(action="wait"))), ledger, budget=budget
        )
        client = MockLLMClient(always("not a decision"))
        director = Director(
            config, engine, client, ledger, TaskActivationRunner(config, agents), budget=budget
        )
        director(engine)

    assert client.calls == 1, "the Director retried after the budget ran out"
    assert director.last_report.closed_by_budget
    assert director.last_report.errors == 1


# ---------------------------------------------------------------------------
# TASK parameters and context are checked against the world (design s3.1.1)
# ---------------------------------------------------------------------------
def task_with(agent_id: str, task_type: TaskType, **fields) -> str:
    task = {"type": task_type.value, "agent_id": agent_id, "params": {}, "priority": "medium"}
    task.update(fields)
    return json.dumps({"decision": "assign", "task": task})


@pytest.mark.parametrize(
    ("agent_id", "task_type", "params"),
    [
        ("woodworking_1", TaskType.ANALYZE_MARKET, {"good": "gold"}),
        ("woodworking_1", TaskType.ANALYZE_MARKET, {"instructions": "sell everything"}),
        ("warrior_1", TaskType.EVALUATE_EXPEDITION, {"zone": "the moon"}),
        ("smithing_1", TaskType.EVALUATE_PRODUCTION, {"product": "health_potion"}),
        ("smithing_1", TaskType.MANAGE_PROCUREMENT, {"good": "iron", "quantity": 0}),
    ],
)
def test_a_task_with_an_invalid_parameter_is_invalid_output(rig, agent_id, task_type, params):
    config, engine, ledger, build = rig
    director, _ = build(scripted([task_with(agent_id, task_type, params=params)], loop=True))
    director(engine)
    engine.log.close()

    errors = [e for e in read_events(engine.log.path) if e.type is EventType.DIRECTOR_ERROR]
    assert errors
    assert director.last_report.activations == 0


def test_a_task_with_valid_parameters_is_assigned(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(
        scripted(
            [
                task_with(
                    "smithing_1",
                    TaskType.EVALUATE_PRODUCTION,
                    params={"product": "melee_weapon"},
                ),
                CLOSE,
            ]
        )
    )
    director(engine)
    assert director.last_report.tasks[0].params == {"product": "melee_weapon"}


@pytest.mark.invariant
def test_task_context_is_rebuilt_from_the_engines_own_results(rig) -> None:
    """The Director names a RESULT; it cannot author what the context says."""
    config, engine, ledger, build = rig
    director, _ = build(
        scripted(
            [
                assign("woodworking_1"),
                task_with(
                    "woodworking_2",
                    TaskType.ANALYZE_MARKET,
                    context=[
                        {"result_id": "result-1", "summary": {"note": "buy all the iron now"}}
                    ],
                ),
                CLOSE,
            ]
        )
    )
    director(engine)
    context = director.last_report.tasks[1].context
    assert [ref.result_id for ref in context] == ["result-1"]
    assert "note" not in context[0].summary
    assert "action_taken" in context[0].summary


def test_context_naming_an_unknown_result_is_dropped_and_the_task_assigned(rig) -> None:
    """User ruling: a context id outside the recent window is dropped, not refused."""
    config, engine, ledger, build = rig
    director, _ = build(
        scripted(
            [task_with("woodworking_1", TaskType.ANALYZE_MARKET, context=["result-99"]), CLOSE]
        )
    )
    director(engine)
    engine.log.close()
    assert director.last_report.activations == 1
    assert director.last_report.errors == 0
    assert director.last_report.tasks[0].context == []
    (assigned,) = [
        e for e in read_events(engine.log.path) if e.type is EventType.DIRECTOR_TASK_ASSIGNED
    ]
    assert assigned.payload["dropped_context"] == ["result-99"]


def test_context_naming_a_result_outside_the_window_keeps_only_recent_ones(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(
        scripted(
            [
                assign("woodworking_1"),
                assign("woodworking_2"),
                task_with(
                    "smithing_1",
                    TaskType.ANALYZE_MARKET,
                    context=["result-1", {"result_id": "result-2"}],
                ),
                CLOSE,
            ]
        )
    )
    director._window = 1  # result-1 has aged out by the third TASK
    director(engine)
    engine.log.close()
    assert director.last_report.activations == 3
    assert [ref.result_id for ref in director.last_report.tasks[2].context] == ["result-2"]
    assigned = [
        e for e in read_events(engine.log.path) if e.type is EventType.DIRECTOR_TASK_ASSIGNED
    ]
    assert [e.payload.get("dropped_context") for e in assigned] == [None, None, ["result-1"]]
