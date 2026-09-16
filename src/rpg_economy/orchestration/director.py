"""Condition A: the centralized Director (design v1.0 s3.1).

The Director sits **outside** the economy. It reads the global summary and the
recent TASK/RESULT history, selects one agent, and assigns it a TASK from the
closed vocabulary. It cannot move gold, inventory, resources, prices or
probabilities, cannot force a transaction or a production run, cannot declare an
expedition's outcome, and cannot reach an agent's private reasoning (s3.1.3).
None of those capabilities is withheld by policy here: they are simply absent
from what this module can do.

**The cap is hard: at most K + 1 LLM calls per macro-cycle, retries included.**
Every call decrements it. Each valid inference does exactly one of two things --
assign one TASK or close the cycle.

The invalid-output policy, exactly as s3.1 states it:

* an invalid decision admits **one** retry, and only while inferences remain
  within the cap;
* if the retry is also invalid, the TASK is discarded and logged as
  ``director_error``;
* **if the cap is exhausted during the retry, no further call is made and the
  macro-cycle closes.**

Director calls do not consume K economic activations in Experiment 1, but they
are logged in full as overhead and they do consume the Experiment 2 token
budget (s8.4).
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Resource, TaskType, TokenBucket, parse_good
from rpg_economy.engine import Engine
from rpg_economy.llm.budget import TokenBudget
from rpg_economy.llm.client import LLMClient, LLMError, ProviderError
from rpg_economy.llm.tokens import TokenLedger
from rpg_economy.logging_ import EventType, prompt_sha
from rpg_economy.logging_.metadata import clear_hash_caches, register_hash_cache
from rpg_economy.orchestration.global_view import build_global_summary, render_global_summary
from rpg_economy.orchestration.protocol import (
    TASK_PARAMS,
    TASK_PURPOSES,
    TASK_TARGET_ROLES,
    DirectorDecision,
    InvalidDirectorOutput,
    Result,
    ResultReference,
    Task,
)


def _task_table() -> str:
    """What each TASK type is for, who may receive it and its parameters (s3.1.1)."""
    lines = []
    for task_type in TaskType:
        params = ", ".join(f"{k} ({v})" for k, v in TASK_PARAMS[task_type].items())
        roles = "/".join(sorted(TASK_TARGET_ROLES[task_type]))
        lines.append(
            f"  {task_type.value}: {TASK_PURPOSES[task_type]}; for {roles}; params: {params}"
        )
    return "\n".join(lines)


DIRECTOR_SYSTEM = (
    "You are the Director of a simulated RPG economy. You coordinate which agent "
    "acts next; you are not a participant.\n\n"
    "You CANNOT alter gold, inventories, resources, prices or probabilities, force "
    "a transaction or a production run, declare an expedition's outcome, or see any "
    "agent's private reasoning. You choose who acts and what task they are given.\n\n"
    "An agent acts only when you task it. Adventurers bring raw materials into the "
    "economy only from expeditions, which they set out on when given "
    "EVALUATE_EXPEDITION; artisans craft only when given a task that lets them "
    "produce. An agent never tasked does nothing. Each agent can be given at most one "
    "task per macro-cycle.\n\n"
    "Each reply does exactly ONE of:\n"
    '  {"decision":"assign","task":{"type":<TASK>,"agent_id":<id>,'
    '"params":{...},"priority":"high"|"medium"|"low","context":[<result_id>,...]}}\n'
    '  {"decision":"close"}\n\n'
    "TASK types, who may receive them, and their optional parameters:\n"
    + _task_table()
    + "\nA good is a raw material or product name, a resource is a raw material, a "
    "zone is a zone id, and a product must be one the assignee's specialization "
    "crafts. context is optional and may only name ids of the recent results you "
    "were shown.\n"
    "Reply with one JSON object and nothing else. Do not deliberate: assign or close."
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


#: Parameters whose values are ids in lower case (goods, zones, products).
_ID_PARAMS = {"good", "resource", "product", "zone"}


def _normalize_decision(payload: dict) -> dict:
    """Tolerate what carries no decision, exactly as the agents' parser does.

    Unknown keys are dropped, the decision and priority are case-folded, the
    TASK type is upper-cased with spaces or hyphens as underscores, id-like
    parameters are normalized as the agents' targets are ("Arcane Crystal"
    names arcane_crystal), a numeric quantity given as a string is read as a
    number, a parameter that is null or blank is dropped, and a close that
    carries an empty or null task is read as a plain close. The
    Director is held to the same standard as the agents: a formatting habit is
    not a coordination failure, and charging it would bias Condition A alone.
    """
    from rpg_economy.agents.parsing import normalize_id

    out: dict = {k: payload[k] for k in ("decision", "task") if k in payload}
    if isinstance(out.get("decision"), str):
        out["decision"] = out["decision"].strip().lower()
    # {"decision": "close", "task": {}} closes: an empty task assigns nothing.
    if out.get("decision") == "close" and out.get("task") in ({}, None):
        out.pop("task", None)
    task = out.get("task")
    if isinstance(task, dict):
        clean = {
            k: task[k] for k in ("type", "agent_id", "params", "priority", "context") if k in task
        }
        if isinstance(clean.get("type"), str):
            clean["type"] = clean["type"].strip().upper().replace(" ", "_").replace("-", "_")
        if isinstance(clean.get("priority"), str):
            clean["priority"] = clean["priority"].strip().lower()
        elif clean.get("priority") is None:
            clean.pop("priority", None)
        params = clean.get("params")
        if params is None:
            clean["params"] = {}
        elif isinstance(params, dict):
            fixed = {}
            for raw_key, value in params.items():
                key = str(raw_key).strip().lower()
                # Every TASK param is optional, so a null or blank one says
                # nothing: it is dropped rather than refused, as an agent's
                # null field is.
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                if isinstance(value, str) and key in _ID_PARAMS:
                    value = normalize_id(value)
                elif isinstance(value, str) and key == "quantity" and value.strip().isdigit():
                    value = int(value.strip())
                elif isinstance(value, float) and value.is_integer():
                    value = int(value)
                fixed[key] = value
            clean["params"] = fixed
        out["task"] = clean
    return out


def director_prompt_sources() -> tuple:
    """Every function whose source shapes the Director's user prompt.

    The builder, the TASK table, and what builds the recent RESULTs the prompt
    serialises.
    """
    return (
        Director._user_prompt,
        _task_table,
        Director._recent_window,
        Director._recent_references,
        Result.to_reference,
    )


def global_view_sources() -> tuple:
    """Every function whose source shapes the global view's text (design s3.3).

    What the Director and the Condition C bulletin read: the builder, the
    renderer and its helpers, including the statistics it prints.
    """
    from rpg_economy.orchestration import global_view

    return (
        global_view.build_global_summary,
        global_view._public_shock,
        global_view.render_global_summary,
        global_view._pairs,
        global_view._cell,
        global_view._mean,
        global_view._stdev,
    )


def prompt_hashes() -> dict[str, str]:
    """Every prompt template's hash: the agents' and the Director's.

    What run metadata records (design v1.0 s15). Computed once per
    process, for the reason ``agents.prompts.prompt_hashes`` is; either
    function's ``cache_clear()`` clears both caches.
    """
    return dict(_prompt_hashes())


@functools.cache
def _prompt_hashes() -> tuple[tuple[str, str], ...]:
    import inspect

    from rpg_economy.agents.prompts import prompt_hashes as agent_prompt_hashes
    from rpg_economy.orchestration import global_view

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def sources(functions: tuple) -> str:
        return "".join(inspect.getsource(f) for f in functions)

    return tuple(
        {
            **agent_prompt_hashes(),
            "director_system": digest(DIRECTOR_SYSTEM),
            "director_prompt_builder": digest(sources(director_prompt_sources())),
            # Every piece that shapes the global view's text: its sources, the
            # public shock details and the format version.
            "global_view_format": digest(
                global_view.GLOBAL_VIEW_VERSION
                + repr(global_view.PUBLIC_SHOCK_DETAILS)
                + sources(global_view_sources())
            ),
        }.items()
    )


# Clearing either module's cache clears both (logging_.metadata.register_hash_cache).
register_hash_cache(_prompt_hashes.cache_clear)
prompt_hashes.cache_clear = clear_hash_caches  # type: ignore[attr-defined]


#: Chooses a TASK given the Director's view. Used by the offline mock.
DirectorResponder = Callable[[str, str], str]


@dataclass
class DirectorCycleReport:
    """What the Director did in one macro-cycle."""

    calls_used: int = 0
    activations: int = 0
    retries: int = 0
    errors: int = 0
    closed_by_decision: bool = False
    closed_by_cap: bool = False
    closed_by_budget: bool = False
    #: The budget was already spent when the cycle opened, as B and C mark it.
    post_exhaustion: bool = False
    k: int | None = None
    tasks: list[Task] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "director_calls": self.calls_used,
            "effective_activations": self.activations,
            "k": self.k,
            "director_retries": self.retries,
            "director_errors": self.errors,
            "closed_by_decision": self.closed_by_decision,
            "closed_by_cap": self.closed_by_cap,
            "closed_by_budget": self.closed_by_budget,
            # A cycle that opened with the budget already spent, as B and C
            # mark theirs: the K-binding rate leaves it out (design s8.4). A
            # cycle in which the budget ran out is not one.
            "post_exhaustion": self.post_exhaustion,
        }


class Director:
    """The centralized coordinator of Condition A."""

    def __init__(
        self,
        config: SimulationConfig,
        engine: Engine,
        client: LLMClient,
        ledger: TokenLedger,
        activate: Callable[[Engine, str, Task], Result],
        *,
        budget: TokenBudget | None = None,
    ) -> None:
        self._budget = budget or TokenBudget()
        self._config = config
        self._engine = engine
        self._client = client
        self._ledger = ledger
        self._activate = activate
        self._recent: list[Result] = []
        # The world's fixed rules sit in the system prompt: they never change
        # within a run, so a provider can serve them from its prompt cache.
        from rpg_economy.agents.prompts import render_world

        self._system = DIRECTOR_SYSTEM + "\n\n" + render_world(config)
        #: Logged in full once, the first time it is sent; calls name it by digest.
        #: System prompts already logged in full this run, by digest.
        self._systems_logged: set[str] = set()
        #: Per agent, how many TASKs it has had this run and the tick of the
        #: last: public, mechanical facts the Director's roster shows.
        self._tasked: dict[str, tuple[int, int | None]] = {}
        #: Agents already tasked in the current macro-cycle: at most one TASK
        #: each, as B and C activate an agent at most once per cycle.
        self._cycle_agents: set[str] = set()
        #: How many recent RESULTs the Director sees and may reference.
        self._window = config.observability.director_recent_results
        self._next_task = 1
        #: Context ids the last parsed decision named outside the recent window.
        self._dropped_context: list[str] = []
        self._last_report = DirectorCycleReport()
        self._cumulative_calls = 0
        #: The cycle's global view, rendered once: (cycle, text).
        self._cycle_view: tuple[int, str] | None = None
        engine.add_cycle_summary_provider(lambda: self._last_report.to_payload())

    @property
    def call_cap(self) -> int:
        """K + 1 LLM calls per macro-cycle, retries included (design s3.1)."""
        return self._config.time.k_activations + 1

    @property
    def last_report(self) -> DirectorCycleReport:
        return self._last_report

    @property
    def recent_results(self) -> list[Result]:
        return list(self._recent)

    # -- the view ----------------------------------------------------------
    def _open_view(self, engine: Engine) -> None:
        """Build the cycle's global view once, as the decision phase opens.

        The moment and the generator of Condition C's bulletin (design s3.3),
        so the two are byte-identical for the same state, and every Director
        call of the cycle reads this one view. Logged in full, as the bulletin
        is, so the run can be reconstructed from its log (s15).
        """
        summary = build_global_summary(self._config, engine)
        text = render_global_summary(summary)
        self._cycle_view = (engine.cycle, text)
        engine.log_event(
            EventType.DIRECTOR_VIEW,
            payload={"summary": summary, "view_sha": _sha(text)},
        )

    def _view(self) -> str:
        """The cycle's global view, from the generator Condition C's bulletin shares.

        Outside a decision phase (a prompt built directly) there is no cycle
        view yet, so a fresh one is rendered without being logged.
        """
        if self._cycle_view is not None and self._cycle_view[0] == self._engine.cycle:
            return self._cycle_view[1]
        return render_global_summary(build_global_summary(self._config, self._engine))

    def _user_prompt(self, report: DirectorCycleReport) -> str:
        # The same compact id:role/specialization record agents get for the
        # directory, marking who is away (design s3.1: the Director sees which
        # agents are available). An adventurer on an expedition can trade but
        # not equip or set out; one resting in town can trade and equip but not
        # set out yet.
        tick = self._engine.tick

        def tasked(agent_id: str) -> str:
            count, last = self._tasked.get(agent_id, (0, None))
            return f"[{count},t{last}]" if count else "[0]"

        # Fewest tasks first, then the longest untasked, then by id. In id
        # order the real Director tasked the head of the list and never reached
        # warrior_* or woodworking_*, which sort last.
        def standing(agent_id: str) -> tuple[int, int, str]:
            count, last = self._tasked.get(agent_id, (0, None))
            return (count, -1 if last is None else last, agent_id)

        agents = " ".join(
            f"{agent_id}:{account.role.value}/{account.specialization or '-'}"
            + ("*" if account.expedition is not None else "~" if account.is_resting(tick) else "")
            + tasked(agent_id)
            for agent_id in sorted(self._engine.state.agent_ids, key=standing)
            for account in (self._engine.state.account(agent_id),)
        )
        # Only structured RESULT fields may reach the Director (s3.1.2).
        recent = [ref.model_dump(mode="json") for ref in self._recent_references()]
        return (
            "Global summary:\n"
            + self._view()
            + "\n\nAgents (id:role/specialization; * = away on an expedition, can trade "
            + "but not equip or set out; ~ = resting in town, can trade and equip but "
            + "not set out yet; [n,tT] = tasks given so far and the tick of the last, "
            + "[0] = never tasked; listed fewest tasks first):\n"
            + agents
            + "\n\nRecent results (each answers the task_id it names):\n"
            + json.dumps(recent, sort_keys=True, separators=(",", ":"), default=str)
            + f"\n\nActivations used this macro-cycle: {report.activations} of "
            + f"{self._config.time.k_activations}."
            + "\nAlready tasked this macro-cycle (cannot be tasked again until the next): "
            + (", ".join(sorted(self._cycle_agents)) or "none")
            + "."
        )

    def _recent_window(self) -> list[Result]:
        """The recent RESULTs the Director sees and may name as context.

        Empty with a window of 0: ``self._recent[-0:]`` would be the whole
        history.
        """
        if self._window <= 0:
            return []
        return self._recent[-self._window :]

    def _recent_references(self) -> list[ResultReference]:
        return [result.to_reference() for result in self._recent_window()]

    # -- one inference -----------------------------------------------------
    def _infer(self, report: DirectorCycleReport) -> DirectorDecision:
        """Make one Director call. Always consumes one unit of the cap."""
        prompt = self._user_prompt(report)
        report.calls_used += 1
        self._cumulative_calls += 1

        try:
            response = self._client.complete(system=self._system, user=prompt)
        except ProviderError:
            raise
        except LLMError as exc:
            self._log_call(report, None, str(exc), prompt)
            raise InvalidDirectorOutput(f"director call failed: {exc}") from exc

        # Parsed before logging, so the call's own record says why an output was
        # invalid -- the first attempt's reason would otherwise be lost when the
        # retry succeeds.
        decision: DirectorDecision | None = None
        failure: InvalidDirectorOutput | None = None
        try:
            decision = self._parse(response.text)
        except InvalidDirectorOutput as exc:
            failure = exc

        # Log the call before charging it, so a call that crosses the cap is
        # recorded ahead of the exhaustion it caused rather than behind it.
        self._log_call(report, response, None if failure is None else str(failure), prompt)
        self._ledger.record(
            input_by_bucket={TokenBucket.DIRECTOR: response.input_tokens},
            output_tokens=response.output_tokens,
            output_bucket=TokenBucket.DIRECTOR,
            authoritative_total=response.total_tokens,
            billed=response.billed,
        )
        if self._budget.charge(
            response.total_tokens, tick=self._engine.tick, cycle=self._engine.cycle
        ):
            self._engine.log_event(EventType.BUDGET_EXHAUSTED, payload=self._budget.snapshot())
        if failure is not None:
            raise failure
        assert decision is not None
        return decision

    def _system_digest(self) -> str:
        """The system prompt's digest, logging its text the first time."""
        sha = prompt_sha(self._system)
        if sha not in self._systems_logged:
            self._systems_logged.add(sha)
            self._engine.log_event(
                EventType.PROMPT_SYSTEM, payload={"sha": sha, "text": self._system}
            )
        return sha

    def _log_call(
        self, report: DirectorCycleReport, response, error: str | None, prompt: str
    ) -> None:
        self._engine.log_event(
            EventType.DIRECTOR_CALL,
            payload={
                "call_in_cycle": report.calls_used,
                "cap": self.call_cap,
                "cap_remaining": self.call_cap - report.calls_used,
                "cumulative_calls": self._cumulative_calls,
                "model": None if response is None else response.model,
                "input_tokens": 0 if response is None else response.input_tokens,
                "output_tokens": 0 if response is None else response.output_tokens,
                "reasoning_tokens": 0 if response is None else response.reasoning_tokens,
                "cached_tokens": 0 if response is None else response.cached_tokens,
                "finish_reason": None if response is None else response.finish_reason,
                # Whether the provider billed this call, as the client reports it.
                "billed": False if response is None else response.billed,
                # What the call saw and said: the RESULTs that motivated it
                # (design s15), a digest of its global view (logged in full once
                # per cycle as director_view), and the raw completion.
                "recent_results": [r.result_id for r in self._recent_window()],
                "view_sha": _sha(self._view()),
                "prompt_chars": len(prompt),
                # The prompt as sent: the system part by digest (logged once as
                # prompt_system), the user part in full.
                "system_sha": self._system_digest(),
                "user": prompt,
                # In full: a run is reconstructed from its log (design s15).
                "raw": None if response is None else response.text,
                "seed": None if response is None else response.seed,
                "error": error,
            },
        )

    def _parse(self, text: str) -> DirectorDecision:
        """Parse a completion into a decision, validating against the vocabulary."""
        from rpg_economy.agents.parsing import extract_json

        try:
            payload = extract_json(text)
        except Exception as exc:
            raise InvalidDirectorOutput(str(exc)) from exc

        payload = _normalize_decision(payload)
        self._dropped_context = []
        task_payload = payload.get("task")
        if isinstance(task_payload, dict):
            # The engine assigns the id and builds the context: the Director
            # names recent RESULTs by id, and their structured summaries come
            # from the engine's own record (s3.1.1), so nothing the Director
            # writes can travel as context.
            task_payload = {
                **task_payload,
                "task_id": f"task-{self._next_task}",
                "context": self._context_from(task_payload.get("context")),
            }
            payload = {**payload, "task": task_payload}

        try:
            decision = DirectorDecision.model_validate(payload)
        except Exception as exc:
            raise InvalidDirectorOutput(f"invalid director decision: {exc}") from exc

        if decision.task is not None:
            self._validate_task(decision.task)
        return decision

    def _context_from(self, raw: object) -> list[dict]:
        """Rebuild TASK context from the ids of recent RESULTs.

        An id outside the recent window (stale, or never issued) is dropped and
        recorded in ``_dropped_context``, and the TASK is still assigned: the
        context is only a reference, so a stale one carries no decision and
        does not make the TASK invalid. A malformed entry is still refused.
        """
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise InvalidDirectorOutput("context must be a list of recent result ids")
        recent = {result.result_id: result for result in self._recent_window()}
        references = []
        for item in raw:
            result_id = item.get("result_id") if isinstance(item, dict) else item
            if not isinstance(result_id, str):
                raise InvalidDirectorOutput(f"context names no RESULT id: {item!r}")
            if result_id not in recent:
                self._dropped_context.append(result_id)
                continue
            references.append(recent[result_id].to_reference().model_dump(mode="json"))
        return references

    def _validate_task(self, task: Task) -> None:
        """Check a TASK against the closed vocabulary, the target's role and the world."""
        if task.agent_id not in self._engine.state.accounts:
            raise InvalidDirectorOutput(f"unknown agent {task.agent_id!r}")
        if task.agent_id in self._cycle_agents:
            raise InvalidDirectorOutput(
                f"{task.agent_id} already has a task this macro-cycle; each agent gets "
                f"at most one per cycle"
            )
        account = self._engine.state.account(task.agent_id)
        role = account.role.value
        allowed = TASK_TARGET_ROLES[task.type]
        if role not in allowed:
            raise InvalidDirectorOutput(
                f"{task.type.value} cannot be assigned to a {role}; it targets {sorted(allowed)}"
            )

        spec = TASK_PARAMS[task.type]
        for key, value in task.params.items():
            kind = spec.get(key)
            if kind is None:
                raise InvalidDirectorOutput(
                    f"{task.type.value} takes no parameter {key!r}; it takes {sorted(spec)}"
                )
            if not self._param_ok(kind, value, account):
                raise InvalidDirectorOutput(f"invalid {key} for {task.type.value}: {value!r}")

    def _param_ok(self, kind: str, value: object, account) -> bool:
        match kind:
            case "quantity":
                return isinstance(value, int) and value > 0
            case "zone":
                return value in self._engine.zones.zone_ids
            case "good" | "resource":
                try:
                    good = parse_good(str(value))
                except ValueError:
                    return False
                return kind == "good" or isinstance(good, Resource)
            case "product":
                specialization = self._engine.production.specialization_of(account)
                return specialization is not None and value in {
                    product.value
                    for product in self._engine.production.products_for(specialization)
                }
        return False  # pragma: no cover -- every kind in TASK_PARAMS is handled

    # -- the macro-cycle ---------------------------------------------------
    def __call__(self, engine: Engine) -> None:
        """Run the Director's decision phase for one macro-cycle."""
        k = self._config.time.k_activations
        report = DirectorCycleReport(k=k)

        # Budget exhausted: all LLM calls cease, the Director's included. The
        # engine keeps running mechanical ticks to the planned horizon (s8.4).
        if not self._budget.may_call():
            report.closed_by_budget = True
            report.post_exhaustion = True
            self._last_report = report
            return

        self._open_view(engine)
        self._cycle_agents = set()

        while report.activations < k:
            if report.calls_used >= self.call_cap:
                report.closed_by_cap = True
                break
            if not self._budget.may_call():
                report.closed_by_budget = True
                break

            try:
                decision = self._infer(report)
            except InvalidDirectorOutput as exc:
                # One retry, and only if the cap still has room for it.
                if report.calls_used >= self.call_cap:
                    # Exhausted during the failure: close with no further call.
                    report.closed_by_cap = True
                    report.errors += 1
                    self._log_error(report, str(exc), retried=False)
                    break
                if not self._budget.may_call():
                    # The failed call spent the last of the budget: no retry,
                    # because every call after exhaustion is refused (s8.4).
                    report.closed_by_budget = True
                    report.errors += 1
                    self._log_error(report, str(exc), retried=False)
                    break
                report.retries += 1
                try:
                    decision = self._infer(report)
                except InvalidDirectorOutput as retry_exc:
                    report.errors += 1
                    self._log_error(report, str(retry_exc), retried=True)
                    continue

            if decision.decision == "close":
                report.closed_by_decision = True
                engine.log_event(
                    EventType.DIRECTOR_CYCLE_CLOSED,
                    payload={"activations": report.activations, "calls": report.calls_used},
                )
                break

            task = decision.task
            assert task is not None  # guaranteed by DirectorDecision validation
            # Set by the parse that produced this decision (the last one made).
            dropped_context = list(self._dropped_context)
            self._next_task += 1

            # The agent's own activation is an LLM call as well. A TASK whose
            # assigning call spent the last of the budget can never be carried
            # out: it is logged as dropped, not as assigned, so no assigned TASK
            # is left without its RESULT.
            if not self._budget.may_call():
                report.closed_by_budget = True
                engine.log_event(
                    EventType.DIRECTOR_TASK_DROPPED,
                    agent_id=task.agent_id,
                    payload={**task.to_payload(), "reason": "budget_exhausted"},
                )
                break

            report.tasks.append(task)
            engine.log_event(
                EventType.DIRECTOR_TASK_ASSIGNED,
                agent_id=task.agent_id,
                payload={
                    **task.to_payload(),
                    # Context ids outside the recent window, dropped rather
                    # than refused. Present only when there were any.
                    **({"dropped_context": dropped_context} if dropped_context else {}),
                },
            )

            self._cycle_agents.add(task.agent_id)
            count, _ = self._tasked.get(task.agent_id, (0, None))
            self._tasked[task.agent_id] = (count + 1, engine.tick)
            result = self._activate(engine, task.agent_id, task)
            self._recent.append(result)
            report.activations += 1
            engine.log_event(
                EventType.RESULT_RETURNED,
                agent_id=task.agent_id,
                payload=result.to_payload(),
            )

        self._last_report = report

    def _log_error(self, report: DirectorCycleReport, reason: str, *, retried: bool) -> None:
        self._engine.log_event(
            EventType.DIRECTOR_ERROR,
            payload={
                "reason": reason,
                "retried": retried,
                "calls_used": report.calls_used,
                "cap": self.call_cap,
                "cap_exhausted": report.calls_used >= self.call_cap,
            },
        )
