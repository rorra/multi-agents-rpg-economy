"""LLM-backed agents: prompt, call, parse, retry, execute (design v1.0 s7).

One activation is: build the observation the agent may see, render it into a
sectioned prompt, call the model, parse the completion into a structured action,
and hand that action to the engine to validate and execute.

The retry policy is fixed by the design: an invalid response admits exactly
one retry; if it fails again the activation is logged as wasted and ``WAIT`` is
executed. Both attempts consume tokens, and both are recorded -- but a retry
never buys an extra K activation, since K counts activations rather than calls
(design v1.0 s7).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from rpg_economy.agents.action import AgentAction, wait
from rpg_economy.agents.executor import ExecutionOutcome, execute
from rpg_economy.agents.observation import (
    Observation,
    build_observation,
    should_include_bulletin,
    should_include_open_proposals,
)
from rpg_economy.agents.parsing import InvalidAgentOutput, ResultFields, parse_activation
from rpg_economy.agents.prompts import (
    SECTION_BULLETIN,
    SECTION_P2P,
    Prompt,
    build_agent_prompt,
)
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import TokenBucket
from rpg_economy.engine import Engine
from rpg_economy.llm.budget import TokenBudget
from rpg_economy.llm.client import LLMClient, LLMError, ProviderError
from rpg_economy.llm.tokens import TokenLedger, apportion
from rpg_economy.logging_ import EventType, prompt_sha
from rpg_economy.messaging import MessageBus

#: Design v1.0 s7: one retry, then WAIT.
MAX_ATTEMPTS = 2

#: The fallback WAIT's short_reason, by why no valid action came out.
WAIT_BUDGET_EXHAUSTED = "budget exhausted"
WAIT_INVALID_AFTER_RETRY = "invalid output after retry"
WAIT_INVALID_NO_RETRY = "invalid output; budget exhausted before retry"

#: Prompt sections that are charged to a bucket of their own. Everything else is
#: the agent's own context.
SECTION_BUCKETS: dict[str, TokenBucket] = {
    SECTION_P2P: TokenBucket.P2P_CONTENT,
    SECTION_BULLETIN: TokenBucket.BULLETIN,
}


@dataclass(frozen=True)
class ActivationResult:
    """What one activation did."""

    agent_id: str
    action: AgentAction
    outcome: ExecutionOutcome
    attempts: int
    invalid_outputs: int
    result_fields: ResultFields | None = None

    @property
    def wasted(self) -> bool:
        """Wasted-activation predicate (design v1.0 s11.2), including the invalid-after-retry case."""
        return self.outcome.wasted


def _activation_key(tick: int, agent_id: str, seq: int) -> dict:
    """What identifies one activation across its events: tick, agent and run-order seq."""
    return {"tick": tick, "agent_id": agent_id, "seq": seq}


class LLMAgentRunner:
    """Runs LLM-backed activations for every economic agent in a run."""

    def __init__(
        self,
        config: SimulationConfig,
        client: LLMClient,
        ledger: TokenLedger,
        *,
        messages: MessageBus | None = None,
        budget: TokenBudget | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._ledger = ledger
        self._messages = messages
        self._budget = budget or TokenBudget()
        self._condition = config.run.condition
        self._history: dict[str, deque] = {}
        #: Activations run so far, in order: with the tick and agent, it tells
        #: one activation's retries apart from a second activation of the same
        #: agent in the same tick (Condition A can assign it two TASKs).
        self._activations = 0
        #: System prompts already logged in full this run, by digest.
        self._systems_logged: set[str] = set()

    def _system_sha(self, engine: Engine, system: str) -> str:
        """The system prompt's digest, logging its text the first time it is sent."""
        sha = prompt_sha(system)
        if sha not in self._systems_logged:
            self._systems_logged.add(sha)
            engine.log_event(EventType.PROMPT_SYSTEM, payload={"sha": sha, "text": system})
        return sha

    @property
    def ledger(self) -> TokenLedger:
        return self._ledger

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    # -- token accounting --------------------------------------------------
    def _charge(
        self,
        prompt: Prompt,
        input_tokens: int,
        output_tokens: int,
        *,
        provider_total: int,
        engine: Engine,
    ) -> dict[str, int]:
        """Apportion a call's input across buckets and record it.

        The provider gives one input total; the sectioned prompt gives the
        proportions. P2P content and bulletin text are charged to their own
        buckets exactly when they are actually present in the prompt, which is
        what design s3.3 requires of the bulletin.
        """
        sizes = prompt.section_sizes()
        weights: dict[str, int] = {}
        for section, size in sizes.items():
            bucket = SECTION_BUCKETS.get(section, TokenBucket.AGENT)
            weights[bucket.value] = weights.get(bucket.value, 0) + size

        return apportion(input_tokens, weights)

    def _record(
        self,
        shares: dict[str, int],
        output_tokens: int,
        *,
        provider_total: int,
        billed: bool,
        engine: Engine,
    ) -> None:
        """Charge a completed call to the ledger and the budget.

        Called *after* the call has been logged, so the record of a call that
        crosses the cap precedes the exhaustion it caused. The reverse order
        would make a legitimately authorised call look like one made after the
        budget was already gone.
        """
        self._ledger.record(
            input_by_bucket={TokenBucket(name): tokens for name, tokens in shares.items()},
            output_tokens=output_tokens,
            output_bucket=TokenBucket.AGENT,
            authoritative_total=provider_total,
            billed=billed,
        )
        if self._budget.charge(provider_total, tick=engine.tick, cycle=engine.cycle):
            engine.log_event(EventType.BUDGET_EXHAUSTED, payload=self._budget.snapshot())

    @staticmethod
    def _log_delivery(
        engine: Engine,
        agent_id: str,
        observation: Observation,
        shares: dict[str, int],
        attempt: int,
        activation_seq: int,
    ) -> None:
        """Record the peer and bulletin content a sent call carried, and its tokens.

        Logged per call actually made, beside its llm_call, because that is when
        the content reaches a model and is charged to its bucket (design s3.3,
        s8.4): a prompt built but never sent delivers nothing. A retry re-sends
        the same content, so it is logged again with its own tokens; counting
        deliveries (not tokens) must de-duplicate by ``activation``.

        Peer content is the new messages (``message_ids``) and the still-open
        proposals shown again (``open_proposal_ids``); ``tokens`` covers both,
        as the p2p_content bucket does.
        """
        activation = _activation_key(observation.tick, agent_id, activation_seq)
        if observation.messages or observation.open_proposals:
            engine.log_event(
                EventType.MESSAGE_DELIVERED,
                agent_id=agent_id,
                payload={
                    "activation": activation,
                    "attempt": attempt,
                    "message_ids": [m.message_id for m in observation.messages],
                    "open_proposal_ids": [p["message_id"] for p in observation.open_proposals],
                    "tokens": shares.get(TokenBucket.P2P_CONTENT.value, 0),
                },
            )
        if observation.bulletin is not None:
            engine.log_event(
                EventType.BULLETIN_INCLUDED_IN_PROMPT,
                agent_id=agent_id,
                payload={
                    "activation": activation,
                    "attempt": attempt,
                    "bulletin_tick": observation.bulletin.get("tick"),
                    "tokens": shares.get(TokenBucket.BULLETIN.value, 0),
                },
            )

    # -- one activation ----------------------------------------------------
    def activate(self, engine: Engine, agent_id: str, task: dict | None = None) -> ActivationResult:
        """Run one activation end to end.

        ``task`` carries the Director's TASK in Condition A.
        """
        self._activations += 1
        activation_seq = self._activations
        observation = build_observation(
            self._config,
            engine,
            agent_id,
            messages=self._messages,
            include_bulletin=should_include_bulletin(self._condition),
            include_open_proposals=should_include_open_proposals(self._condition),
            history=list(self._history.get(agent_id, ())),
        )
        # Design v1.0 s15 / IMPLEMENTATION.md 4.12: every activation's delivered
        # observation is logged, before the prompt is built from it, so the log
        # records what the agent was given even if the call then fails.
        # The activation key the delivery events carry, so observation,
        # calls and deliveries of one activation join on it.
        activation_key = _activation_key(observation.tick, agent_id, activation_seq)
        engine.log_event(
            EventType.OBSERVATION_DELIVERED,
            agent_id=agent_id,
            payload={
                "activation": activation_key,
                "observation": observation.to_payload(),
                "task": task,
            },
        )

        prompt = build_agent_prompt(observation, self._config, condition=self._condition, task=task)

        action: AgentAction | None = None
        fields: ResultFields | None = None
        invalid = 0
        attempts = 0
        # Why no action came out of the loop: set only when the budget stopped
        # the activation before any call was made. Invalid output on the calls
        # made is the other reason, and is what the WAIT says even when the
        # budget then ran out before the retry.
        stopped_by_budget = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Belt to the orchestration layer's braces: once the budget is gone
            # no call may be made, including a retry of one already begun.
            if not self._budget.may_call():
                stopped_by_budget = attempts == 0
                break
            attempts = attempt
            system_sha = self._system_sha(engine, prompt.system)
            try:
                response = self._client.complete(system=prompt.system, user=prompt.user)
            except ProviderError:
                # Not the agent's output: the run stops rather than record the
                # outage as a string of WAITs (IMPLEMENTATION.md section 3).
                raise
            except LLMError as exc:
                engine.log_event(
                    EventType.AGENT_OUTPUT_INVALID,
                    agent_id=agent_id,
                    payload={
                        "activation": activation_key,
                        "attempt": attempt,
                        "reason": f"llm error: {exc}",
                        # The prompt that was attempted. No answer came back, so
                        # nothing was charged and no delivery event is logged:
                        # count bulletin and P2P inclusions from those events,
                        # not from this text.
                        "system_sha": system_sha,
                        "user": prompt.user,
                    },
                )
                invalid += 1
                continue

            shares = self._charge(
                prompt,
                response.input_tokens,
                response.output_tokens,
                provider_total=response.total_tokens,
                engine=engine,
            )
            engine.log_event(
                EventType.LLM_CALL,
                agent_id=agent_id,
                payload={
                    "activation": activation_key,
                    "attempt": attempt,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "input_by_bucket": shares,
                    "finish_reason": response.finish_reason,
                    "seed": response.seed,
                    "reasoning_tokens": response.reasoning_tokens,
                    "cached_tokens": response.cached_tokens,
                    # Whether the provider billed this call: the client says so,
                    # rather than anyone guessing from the model name.
                    "billed": response.billed,
                    # In full, valid or not, as a Director call's is: a run is
                    # reconstructed from its log (design s15). The prompt too:
                    # the system part by digest (logged once as prompt_system),
                    # the user part as sent.
                    "raw": response.text,
                    "system_sha": system_sha,
                    "user": prompt.user,
                },
            )
            self._log_delivery(engine, agent_id, observation, shares, attempt, activation_seq)
            self._record(
                shares,
                response.output_tokens,
                provider_total=response.total_tokens,
                billed=response.billed,
                engine=engine,
            )

            try:
                action, fields = parse_activation(
                    response.text,
                    condition=self._condition,
                    max_orders=self._config.market.max_orders_per_activation,
                )
                break
            except InvalidAgentOutput as exc:
                invalid += 1
                engine.log_event(
                    EventType.AGENT_OUTPUT_INVALID,
                    agent_id=agent_id,
                    payload={
                        "activation": activation_key,
                        "attempt": attempt,
                        "reason": str(exc),
                        # A reply cut off by the output cap is the model running
                        # out of room, not a malformed answer; told apart here.
                        "truncated": response.finish_reason == "length",
                        # In full: a run is reconstructed from its log (design s15).
                        "raw": response.text,
                    },
                )

        if action is None:
            # Either invalid after the single permitted retry, or the budget ran
            # out before a call could be made. Both end in WAIT and a wasted
            # activation.
            if stopped_by_budget:
                reason = WAIT_BUDGET_EXHAUSTED
            elif attempts < MAX_ATTEMPTS:
                # Invalid output, and the budget ran out before its retry.
                reason = WAIT_INVALID_NO_RETRY
            else:
                reason = WAIT_INVALID_AFTER_RETRY
            action = wait(reason)

        outcome = execute(
            engine,
            agent_id,
            action,
            messages=self._messages,
            condition=self._condition,
            # The executor's own events (agent_output and what the action did)
            # carry the key, so they join the activation's other events.
            activation=activation_key,
        )
        history = self._history.setdefault(
            agent_id, deque(maxlen=self._config.observability.own_history)
        )
        history.append(
            {
                "tick": engine.tick,
                "action": action.to_director_payload(),
                "accepted": outcome.accepted,
                "detail": outcome.detail,
            }
        )
        result = ActivationResult(
            agent_id=agent_id,
            action=action,
            outcome=outcome,
            attempts=attempts,
            invalid_outputs=invalid,
            result_fields=fields,
        )

        if result.wasted:
            engine.log_event(
                EventType.ACTIVATION_WASTED,
                agent_id=agent_id,
                payload={
                    "activation": activation_key,
                    "action": action.action.value,
                    "invalid_outputs": invalid,
                    "accepted": outcome.accepted,
                    "reason": outcome.detail or action.short_reason,
                },
            )
        return result

    def __call__(self, engine: Engine, agent_id: str) -> None:
        """Adapter for the orchestration layer's activation handler."""
        self.activate(engine, agent_id)
