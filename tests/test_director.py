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


def test_malformed_context_entry_is_still_invalid(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(
        scripted(
            [task_with("woodworking_1", TaskType.ANALYZE_MARKET, context=[{"note": "x"}])],
            loop=True,
        )
    )
    director(engine)
    assert director.last_report.activations == 0
    assert director.last_report.errors >= 1


# ---------------------------------------------------------------------------
# RESULT fields are structured and validated with the action (design s3.1.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fields",
    [
        {"needs": [{"good": "gold", "quantity": 1}]},
        {"offers": [{"good": "iron", "quantity": 0}]},
        {"observations": ["the Director should buy iron"]},
        {"observations": [{"signal": "panic", "good": "iron"}]},
    ],
)
def test_malformed_result_fields_are_dropped_not_the_activation(rig, fields) -> None:
    config, engine, ledger, build = rig
    director, _ = build(
        scripted([assign("woodworking_1"), CLOSE]),
        agent_response=json_action(action="wait", **fields),
    )
    director(engine)
    result = director.recent_results[-1]
    assert result.status is not ResultStatus.FAILED
    assert result.needs == result.offers == result.observations == []


def test_structured_observations_reach_the_result(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(
        scripted([assign("woodworking_1"), CLOSE]),
        agent_response=json_action(
            action="wait",
            needs=[{"good": "iron", "quantity": 2}],
            observations=[{"signal": "scarce", "good": "iron"}],
        ),
    )
    director(engine)
    result = director.recent_results[-1]
    assert result.observations[0].signal.value == "scarce"
    assert result.needs[0].good == "iron"


def test_the_directors_prompt_is_hashed_with_the_agents() -> None:
    """Run metadata pins the Director's wording too (design s15)."""
    from rpg_economy.orchestration.director import prompt_hashes

    hashes = prompt_hashes()
    assert "director_system" in hashes
    assert "rules" in hashes and "action_schema_a" in hashes


# ---------------------------------------------------------------------------
# A RESULT reports a refusal as a closed code, never the engine's words (s3.1.2)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_refused_action_reaches_the_result_as_a_code(rig) -> None:
    """The engine's message names agents and quotes their holdings (design s5.2)."""
    config, engine, ledger, build = rig
    director, _ = build(
        scripted([assign("woodworking_1"), CLOSE], loop=True),
        agent_response=json_action(action="sell", target="iron", quantity=99, price=10.0),
    )
    director(engine)
    engine.log.close()

    events = list(read_events(engine.log.path))
    rejected = next(e for e in events if e.type is EventType.ACTION_REJECTED)
    assert rejected.payload["code"] == "insufficient_inventory"
    assert "woodworking_1" in rejected.payload["reason"]

    result = next(e for e in events if e.type is EventType.RESULT_RETURNED)
    outcome = result.payload["outcome"]
    assert outcome["rejection"] == "insufficient_inventory"
    assert "detail" not in outcome
    assert rejected.payload["reason"] not in json.dumps(result.payload)

    reference = director.recent_results[-1].to_reference()
    assert rejected.payload["reason"] not in json.dumps(reference.model_dump(mode="json"))


def test_an_accepted_action_reports_no_rejection(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(scripted([assign("woodworking_1"), CLOSE], loop=True))
    director(engine)
    assert director.recent_results[-1].outcome["rejection"] is None


# ---------------------------------------------------------------------------
# The Director's output is read with the agents' leniency
# ---------------------------------------------------------------------------
def test_formatting_that_carries_no_decision_is_normalized() -> None:
    from rpg_economy.orchestration.director import _normalize_decision

    normalized = _normalize_decision(
        {
            "decision": " ASSIGN ",
            "rationale": "free text the protocol has no room for",
            "task": {
                "type": "manage-procurement",
                "agent_id": "smithing_1",
                "priority": "HIGH",
                "params": {"good": " IRON ", "quantity": " 3 ", "zone": "Iron_Marches"},
                "notes": "dropped",
            },
        }
    )

    assert normalized == {
        "decision": "assign",
        "task": {
            "type": "MANAGE_PROCUREMENT",
            "agent_id": "smithing_1",
            "priority": "high",
            "params": {"good": "iron", "quantity": 3, "zone": "iron_marches"},
        },
    }


def test_missing_params_and_priority_are_left_to_the_defaults() -> None:
    from rpg_economy.orchestration.director import _normalize_decision

    normalized = _normalize_decision(
        {"decision": "assign", "task": {"type": "analyze market", "params": None, "priority": None}}
    )
    assert normalized["task"] == {"type": "ANALYZE_MARKET", "params": {}}


def test_a_whole_number_float_is_read_as_an_integer() -> None:
    from rpg_economy.orchestration.director import _normalize_decision

    normalized = _normalize_decision(
        {"decision": "assign", "task": {"type": "X", "params": {"quantity": 2.0}}}
    )
    assert normalized["task"]["params"]["quantity"] == 2
    assert isinstance(normalized["task"]["params"]["quantity"], int)


def test_a_loosely_formatted_task_is_assigned(rig) -> None:
    """A formatting habit is not a coordination failure, and charging it would
    bias Condition A alone."""
    config, engine, ledger, build = rig
    loose = json.dumps(
        {
            "decision": "Assign",
            "reasoning": "smithing needs to move",
            "task": {
                "type": "evaluate-production",
                "agent_id": "smithing_1",
                "params": {"product": "Melee_Weapon"},
                "priority": "HIGH",
                "extra": 1,
            },
        }
    )
    director, _ = build(scripted([loose, CLOSE]))
    director(engine)

    assert director.last_report.errors == 0
    task = director.last_report.tasks[0]
    assert task.type is TaskType.EVALUATE_PRODUCTION
    assert task.params == {"product": "melee_weapon"}
    assert task.priority is Priority.HIGH


def test_a_close_in_capitals_closes_the_cycle(rig) -> None:
    config, engine, ledger, build = rig
    director, client = build(scripted([json.dumps({"decision": "CLOSE"})], loop=True))
    director(engine)
    assert client.calls == 1
    assert director.last_report.closed_by_decision


def test_a_capitalized_param_name_still_normalizes_its_value() -> None:
    from rpg_economy.orchestration.director import _normalize_decision

    out = _normalize_decision(
        {"task": {"type": "evaluate_production", "params": {"Good": "IRON", "Quantity": "3"}}}
    )
    assert out["task"]["params"] == {"good": "iron", "quantity": 3}


def test_the_director_sees_who_is_away_and_who_is_resting_in_town(rig) -> None:
    """Away on a trip an adventurer can trade but not equip or set out; resting, it can trade."""
    from rpg_economy.orchestration.director import DirectorCycleReport

    config, engine, _, build = rig
    director, _ = build(always(assign("warrior_1")))
    engine.start_expedition("warrior_1", "green_hollow")
    ranger = engine.state.account("ranger_1")
    ranger.recovering_until_tick = engine.tick + 5  # back, resting

    prompt = director._user_prompt(DirectorCycleReport())
    assert "* = away on an expedition, can trade but not equip or set out" in prompt
    assert "can only wait" not in prompt
    assert "warrior_1:adventurer/warrior*[0]" in prompt
    assert "ranger_1:adventurer/ranger~[0]" in prompt
    assert "mage_1:adventurer/mage[0]" in prompt
    # Who has been tasked, how often and when: the roster shows it.
    assert "[n,tT] = tasks given so far" in prompt


def test_a_partly_taken_place_orders_reports_each_order_to_the_director(rig) -> None:
    import json as _json

    config, engine, ledger, build = rig
    response = _json.dumps(
        {
            "action": "place_orders",
            "orders": [
                {"action": "buy", "target": "health_potion", "quantity": 1, "price": 4},
                {"action": "sell", "target": "iron", "quantity": 5, "price": 10},
            ],
            "short_reason": "sell what I hold",
        }
    )
    director, _ = build(scripted([assign("warrior_1"), CLOSE], loop=True), agent_response=response)
    director(engine)

    result = director.recent_results[-1]
    assert result.status.value == "PARTIAL"
    orders = result.outcome["orders"]
    assert [o["accepted"] for o in orders] == [True, False]
    assert orders[1]["rejection"] == "insufficient_inventory"
    assert "short_reason" not in _json.dumps(result.action_taken)


# ---------------------------------------------------------------------------
# One global view per macro-cycle, shared with the bulletin (design s3.3, s15)
# ---------------------------------------------------------------------------
def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@pytest.mark.invariant
def test_every_director_call_in_a_cycle_reads_the_same_view(rig) -> None:
    """Built once as the decision phase opens, as the bulletin is, and reused.

    The agents trade between the Director's calls, so a view rebuilt per call
    would change within the cycle; the test checks that it did change.
    """
    from decimal import Decimal

    from rpg_economy.domain import Resource
    from rpg_economy.orchestration.global_view import (
        build_global_summary,
        render_global_summary,
    )

    config, engine, ledger, build = rig
    engine.state.account("warrior_1").receive(Resource.IRON, 5)
    sell = json_action(action="sell", target="iron", quantity=1, price=Decimal("10"))
    director, client = build(
        scripted([assign("warrior_1"), assign("warrior_1"), assign("warrior_1"), CLOSE]),
        agent_response=sell,
    )

    # What the bulletin would carry in Condition C at this moment.
    opening = render_global_summary(build_global_summary(config, engine))
    director(engine)
    after = render_global_summary(build_global_summary(config, engine))
    engine.log.close()

    assert after != opening, "nothing changed within the cycle; the test proves nothing"
    events = list(read_events(engine.log.path))
    views = [e for e in events if e.type is EventType.DIRECTOR_VIEW]
    assert len(views) == 1
    assert render_global_summary(views[0].payload["summary"]) == opening
    assert views[0].payload["view_sha"] == _sha(opening)

    calls = [e for e in events if e.type is EventType.DIRECTOR_CALL]
    assert len(calls) == 4
    assert {e.payload["view_sha"] for e in calls} == {_sha(opening)}
    director_prompts = [user for _, user in client.prompts]
    assert all(opening in user for user in director_prompts)


def test_the_view_is_logged_once_per_cycle(rig) -> None:
    config, engine, ledger, build = rig
    director, _ = build(scripted([assign("woodworking_1"), CLOSE], loop=True))
    for _ in range(3):
        engine.run_cycle(director)
    engine.log.close()

    views = [e for e in read_events(engine.log.path) if e.type is EventType.DIRECTOR_VIEW]
    assert [e.cycle for e in views] == [0, 1, 2]


def test_the_director_completion_is_logged_in_full(rig) -> None:
    config, engine, ledger, build = rig
    long_invalid = "not a decision " * 100
    director, _ = build(always(long_invalid))
    director(engine)
    engine.log.close()

    calls = [e for e in read_events(engine.log.path) if e.type is EventType.DIRECTOR_CALL]
    assert calls and all(e.payload["raw"] == long_invalid for e in calls)


def test_the_task_table_states_each_task_purpose() -> None:
    from rpg_economy.orchestration.director import DIRECTOR_SYSTEM
    from rpg_economy.orchestration.protocol import TASK_PURPOSES

    assert set(TASK_PURPOSES) == set(TaskType)
    for task_type, purpose in TASK_PURPOSES.items():
        assert f"{task_type.value}: {purpose}; for " in DIRECTOR_SYSTEM


# ---------------------------------------------------------------------------
# Experiment 2: budget exhaustion in the Director loop (design s8.4)
# ---------------------------------------------------------------------------
def budget_rig(tmp_path: Path, cap: int, director_script, agent_response=None):
    from rpg_economy.llm.budget import TokenBudget

    config = make_config(run={"condition": "A"})
    log = EventLogWriter(tmp_path / "budget.jsonl", make_metadata(config)).__enter__()
    engine = Engine(config, build_world(config), log)
    ledger = TokenLedger()
    budget = TokenBudget(cap=cap)
    agents = LLMAgentRunner(
        config,
        MockLLMClient(always(agent_response or json_action(action="wait"))),
        ledger,
        budget=budget,
    )
    client = MockLLMClient(director_script)
    director = Director(
        config, engine, client, ledger, TaskActivationRunner(config, agents), budget=budget
    )
    return engine, director, client, log


@pytest.mark.invariant
def test_a_task_whose_assigning_call_spends_the_budget_is_logged_as_dropped(
    tmp_path: Path,
) -> None:
    """Never logged as assigned with no RESULT to follow it."""
    engine, director, client, log = budget_rig(tmp_path, 10, always(assign("woodworking_1")))
    director(engine)
    log.close()

    events = list(read_events(log.path))
    kinds = [e.type for e in events]
    assert client.calls == 1
    assert EventType.DIRECTOR_TASK_ASSIGNED not in kinds
    assert EventType.RESULT_RETURNED not in kinds
    dropped = [e for e in events if e.type is EventType.DIRECTOR_TASK_DROPPED]
    assert len(dropped) == 1
    assert dropped[0].agent_id == "woodworking_1"
    assert dropped[0].payload["reason"] == "budget_exhausted"
    assert dropped[0].payload["type"] == TaskType.ANALYZE_MARKET.value
    assert director.last_report.tasks == []
    assert director.last_report.closed_by_budget


def test_post_exhaustion_marks_only_a_cycle_that_opened_with_the_budget_spent(
    tmp_path: Path,
) -> None:
    """Emitted as B and C emit it, so the K-binding rate leaves it out for A too.

    A cycle in which the budget ran out -- even on the Director's first call,
    leaving it without activations -- is not post-exhaustion: B marks only a
    cycle whose decision phase opened with the budget already spent.
    """
    from rpg_economy.orchestration.director import DirectorCycleReport

    engine, director, _, log = budget_rig(tmp_path, 10, always(assign("woodworking_1")))
    director(engine)  # the Director's own call spends the budget
    report = director.last_report
    assert report.closed_by_budget and report.activations == 0
    assert report.to_payload()["post_exhaustion"] is False
    director(engine)  # exhausted before the cycle opens
    assert director.last_report.to_payload()["post_exhaustion"] is True
    log.close()

    assert DirectorCycleReport(k=2).to_payload()["post_exhaustion"] is False
    spent_late = DirectorCycleReport(k=2, activations=1, closed_by_budget=True)
    assert spent_late.to_payload()["post_exhaustion"] is False


# ---------------------------------------------------------------------------
# TASK context: the Director reads full RESULTs, the assignee public facts only
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_task_context_shows_the_assignee_only_public_facts(tmp_path: Path) -> None:
    """Another agent's rejection codes, needs and offers never reach the assignee.

    They describe that agent's gold and inventory, which no agent sees in B or
    C (design s5.2). The Director still reads the full RESULT.
    """
    config = make_config(run={"condition": "A"})
    prompts: dict[str, list[str]] = {}

    def agent(system: str, user: str) -> str:
        from rpg_economy.agents.offline import agent_id_from_prompt

        prompts.setdefault(agent_id_from_prompt(user), []).append(user)
        return json.dumps(
            {
                "action": "sell",
                "target": "iron",
                "quantity": 99,
                "price": 10,
                "short_reason": "",
                "needs": [{"good": "iron", "quantity": 7}],
                "offers": [{"good": "melee_weapon", "quantity": 3, "price": 40}],
                "observations": [{"signal": "scarce", "good": "iron"}],
            }
        )

    director_script = scripted(
        [
            assign("woodworking_1"),
            task_with("woodworking_2", TaskType.ANALYZE_MARKET, context=["result-1"]),
            CLOSE,
        ]
    )
    with EventLogWriter(tmp_path / "run.jsonl", make_metadata(config)) as log:
        engine = Engine(config, build_world(config), log)
        ledger = TokenLedger()
        agents = LLMAgentRunner(config, MockLLMClient(agent), ledger)
        director = Director(
            config,
            engine,
            MockLLMClient(director_script),
            ledger,
            TaskActivationRunner(config, agents),
        )
        director(engine)

    events = list(read_events(log.path))
    first = next(e for e in events if e.type is EventType.RESULT_RETURNED)
    assert first.payload["outcome"]["rejection"] == "insufficient_inventory"
    assert first.payload["needs"] and first.payload["offers"]

    # The Director's record of the TASK carries the full summary.
    assigned = [e for e in events if e.type is EventType.DIRECTOR_TASK_ASSIGNED]
    assert assigned[1].payload["context"][0]["summary"]["outcome"]["rejection"]

    # The assignee's observation and prompt carry only who, what and how it ended.
    delivered = next(
        e
        for e in events
        if e.type is EventType.OBSERVATION_DELIVERED and e.agent_id == "woodworking_2"
    )
    (reference,) = delivered.payload["task"]["context"]
    assert reference == {
        "result_id": "result-1",
        "agent_id": "woodworking_1",
        "action": "sell",
        "target": "iron",
        "status": "BLOCKED",
    }
    (prompt,) = prompts["woodworking_2"]
    for private in ("insufficient_inventory", "rejection", "melee_weapon", "scarce", '"needs"'):
        assert private not in prompt.split("Your state:")[0], private


def test_a_result_reference_has_a_public_view() -> None:
    result = Result(
        result_id="result-9",
        task_id="task-9",
        agent_id="smithing_1",
        status=ResultStatus.BLOCKED,
        action_taken={"action": "buy", "target": "iron", "quantity": 5, "price": "9"},
        outcome={"accepted": False, "rejection": "insufficient_gold"},
        needs=[Need(good="iron", quantity=5)],
    )
    task = Task(
        task_id="task-10",
        type=TaskType.ANALYZE_MARKET,
        agent_id="woodworking_1",
        context=[result.to_reference()],
    )
    assert task.to_payload()["context"][0]["summary"]["outcome"]["rejection"]
    view = task.to_agent_payload()
    assert view["context"] == [
        {
            "result_id": "result-9",
            "agent_id": "smithing_1",
            "action": "buy",
            "target": "iron",
            "status": "BLOCKED",
        }
    ]
    assert "insufficient_gold" not in json.dumps(view)
    assert {k: v for k, v in view.items() if k != "context"} == {
        k: v for k, v in task.to_payload().items() if k != "context"
    }


# ---------------------------------------------------------------------------
# A null or blank optional param carries no decision
# ---------------------------------------------------------------------------
def test_null_or_blank_params_are_dropped() -> None:
    from rpg_economy.orchestration.director import _normalize_decision

    normalized = _normalize_decision(
        {
            "decision": "assign",
            "task": {
                "type": "EVALUATE_EXPEDITION",
                "params": {"resource": None, "zone": "  ", "Good": ""},
            },
        }
    )
    assert normalized["task"]["params"] == {}


def test_a_task_with_a_null_optional_param_is_assigned(rig) -> None:
    config, engine, ledger, build = rig
    director, client = build(
        scripted(
            [
                task_with("warrior_1", TaskType.EVALUATE_EXPEDITION, params={"zone": None}),
                CLOSE,
            ]
        )
    )
    director(engine)
    engine.log.close()

    assert director.last_report.errors == 0
    assert director.last_report.retries == 0
    (task,) = director.last_report.tasks
    assert task.params == {}


# ---------------------------------------------------------------------------
# The global view's format hash covers everything that shapes its text
# ---------------------------------------------------------------------------
def test_the_global_view_hash_covers_its_version_and_public_details(monkeypatch) -> None:
    from rpg_economy.orchestration import global_view
    from rpg_economy.orchestration.director import prompt_hashes

    prompt_hashes.cache_clear()
    try:
        before = prompt_hashes()["global_view_format"]
        monkeypatch.setattr(global_view, "GLOBAL_VIEW_VERSION", "9.9")
        prompt_hashes.cache_clear()
        bumped = prompt_hashes()["global_view_format"]
        assert bumped != before
        monkeypatch.setattr(global_view, "PUBLIC_SHOCK_DETAILS", ("target_resource",))
        prompt_hashes.cache_clear()
        assert prompt_hashes()["global_view_format"] != bumped
    finally:
        monkeypatch.undo()
        prompt_hashes.cache_clear()


# ---------------------------------------------------------------------------
# A window of zero recent RESULTs shows none and admits none as context
# ---------------------------------------------------------------------------
def test_a_zero_window_shows_no_recent_results_and_admits_no_context(rig) -> None:
    """``recent[-0:]`` is the whole history; a window of 0 must be none of it."""
    config, engine, ledger, build = rig
    prompts: list[str] = []
    replies = iter(
        [
            assign("woodworking_1"),
            task_with("woodworking_2", TaskType.ANALYZE_MARKET, context=["result-1"]),
            CLOSE,
        ]
    )

    def respond(system: str, user: str) -> str:
        prompts.append(user)
        return next(replies)

    director, _ = build(respond)
    director._window = 0
    director(engine)
    engine.log.close()

    assert director.last_report.activations == 2
    assert director.last_report.tasks[1].context == []
    assert director._recent_references() == []
    assert all("Recent results (each answers the task_id it names):\n[]" in p for p in prompts)
    events = list(read_events(engine.log.path))
    calls = [e for e in events if e.type is EventType.DIRECTOR_CALL]
    assert all(call.payload["recent_results"] == [] for call in calls)
    assigned = [e for e in events if e.type is EventType.DIRECTOR_TASK_ASSIGNED]
    assert assigned[1].payload["dropped_context"] == ["result-1"]
