"""The TASK and RESULT protocol, Condition A only (design v1.0 s3.1.1, s3.1.2).

TASK is the Director's instruction: a type from a **closed vocabulary**, its
parameters, a priority, and context that may contain **only references to recent
RESULTs** -- never free-text instructions that would micro-manage the agent
(s3.1.1).

RESULT is what comes back. Its structured fields are the *only* channel from an
agent to the Director (s3.1.2), and `action_taken` carries the structured action
**stripped of `short_reason`**.

Who fills what:

* ``status`` and ``outcome`` are **engine-derived**. An agent does not get to
  declare its own success -- s3.1.3 forbids even the Director from declaring an
  expedition's outcome, and s3.1.2 defines `outcome` as the effect "validado por
  el motor".
* ``needs``, ``offers`` and ``observations`` are **agent-supplied**, optional,
  and empty is valid.

That split means one agent call per activation rather than two: the activation's
structured output carries the action together with its needs and offers, and the
engine supplies what only the engine can know. An agent output that fails
validation takes the ordinary one-retry-then-WAIT path, which is also the RESULT
retry s3.1.2 specifies; on persistent failure the activation is wasted and the
RESULT is ``FAILED``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rpg_economy.domain import ObservationSignal, Priority, ResultStatus, TaskType, parse_good

#: Which roles each TASK type may be assigned to (design v1.0 s3.1.1).
TASK_TARGET_ROLES: dict[TaskType, frozenset[str]] = {
    TaskType.ANALYZE_MARKET: frozenset({"adventurer", "artisan"}),
    TaskType.EVALUATE_EXPEDITION: frozenset({"adventurer"}),
    TaskType.EVALUATE_PRODUCTION: frozenset({"artisan"}),
    TaskType.MANAGE_PROCUREMENT: frozenset({"artisan"}),
    TaskType.MANAGE_INVENTORY: frozenset({"adventurer", "artisan"}),
    TaskType.ADDRESS_IMBALANCE: frozenset({"adventurer", "artisan"}),
}


#: What each TASK type is for, one line each, as the design's TASK table states
#: it (design v1.0 s3.1.1). Shown to the Director and to the assignee.
TASK_PURPOSES: dict[TaskType, str] = {
    TaskType.ANALYZE_MARKET: (
        "evaluate supply, demand and recent prices of the good, and act on what you "
        "find: buy, sell or adjust your orders"
    ),
    TaskType.EVALUATE_EXPEDITION: "decide whether to go on an expedition, zone and risk",
    TaskType.EVALUATE_PRODUCTION: "decide whether to produce, how much and at what cost",
    TaskType.MANAGE_PROCUREMENT: "obtain an input via the market or mediated coordination",
    TaskType.MANAGE_INVENTORY: "sell, keep or re-price inventory",
    TaskType.ADDRESS_IMBALANCE: "respond from the role to a scarcity or excess",
}


#: The parameters each TASK type admits, and what kind of value each takes
#: (design v1.0 s3.1.1). All optional; anything else is invalid output. The
#: kinds are checked against the world: a good must exist, a zone must be one of
#: the zones, a product must be within the assignee's specialization.
TASK_PARAMS: dict[TaskType, dict[str, str]] = {
    TaskType.ANALYZE_MARKET: {"good": "good"},
    TaskType.EVALUATE_EXPEDITION: {"resource": "resource", "zone": "zone"},
    TaskType.EVALUATE_PRODUCTION: {"product": "product"},
    TaskType.MANAGE_PROCUREMENT: {"good": "good"},
    TaskType.MANAGE_INVENTORY: {"good": "good"},
    TaskType.ADDRESS_IMBALANCE: {"good": "good"},
}


def _known_good(value: str) -> str:
    try:
        parse_good(value)
    except ValueError:
        raise ValueError(f"unknown good {value!r}") from None
    return value


class StrictProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResultReference(StrictProtocolModel):
    """A pointer to a recent RESULT, usable as TASK context.

    Identifier plus structured summary only. This is the whole of what s3.1.1
    permits as context: free text here would let the Director issue instructions
    outside the closed vocabulary, which is the thing the vocabulary exists to
    prevent.
    """

    result_id: str
    #: The TASK this RESULT answers, so the Director can read it as an outcome
    #: of its own assignment (design s3.1).
    task_id: str | None = None
    agent_id: str
    status: ResultStatus
    summary: dict[str, Any] = Field(default_factory=dict)

    def public_view(self) -> dict[str, Any]:
        """What an economic agent may read of this RESULT: id, agent, action, status.

        No rejection code, order outcome, need, offer or observation: those
        reveal the acting agent's gold and inventory.
        """
        action = self.summary.get("action_taken") or {}
        return {
            "result_id": self.result_id,
            "agent_id": self.agent_id,
            "action": action.get("action"),
            "target": action.get("target"),
            "status": self.status.value,
        }


class Task(StrictProtocolModel):
    """One Director instruction."""

    task_id: str
    type: TaskType
    agent_id: str
    params: dict[str, str | int] = Field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    context: list[ResultReference] = Field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "type": self.type.value,
            "agent_id": self.agent_id,
            "params": dict(self.params),
            "priority": self.priority.value,
            "context": [ref.model_dump(mode="json") for ref in self.context],
        }

    def to_agent_payload(self) -> dict[str, Any]:
        """The TASK as its assignee reads it: context reduced to public facts.

        The Director reads full RESULTs, but a referenced RESULT may belong to
        another agent, and its rejection codes, order outcomes, needs and offers
        describe that agent's gold and holdings, which no agent sees in B or C
        (design s5.2). The assignee gets only who acted, what it attempted and
        how it ended.
        """
        return {**self.to_payload(), "context": [ref.public_view() for ref in self.context]}


class Need(StrictProtocolModel):
    """A detected need that may motivate a further TASK (design s3.1.2)."""

    good: str
    quantity: int = Field(gt=0)
    priority: Priority = Priority.MEDIUM

    _good = field_validator("good")(_known_good)


class Offer(StrictProtocolModel):
    """A good or capacity the agent can offer."""

    good: str
    quantity: int = Field(gt=0)
    price: Decimal | None = Field(default=None, gt=0, allow_inf_nan=False)

    _good = field_validator("good")(_known_good)


class ResultObservation(StrictProtocolModel):
    """A structured market signal about one good (design s3.1.2)."""

    signal: ObservationSignal
    good: str

    _good = field_validator("good")(_known_good)


class Result(StrictProtocolModel):
    """The structured response to a TASK. The only agent-to-Director channel."""

    result_id: str
    task_id: str
    agent_id: str
    status: ResultStatus
    action_taken: dict[str, Any]
    outcome: dict[str, Any] = Field(default_factory=dict)
    needs: list[Need] = Field(default_factory=list)
    offers: list[Offer] = Field(default_factory=list)
    observations: list[ResultObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_free_text_reasoning(self) -> Result:
        """`short_reason` must never travel to the Director (s3.1.2)."""
        if "short_reason" in self.action_taken:
            raise ValueError(
                "action_taken must not carry short_reason: the Director may not "
                "reach an agent's private reasoning (design v1.0 s3.1.3)"
            )
        return self

    def to_reference(self) -> ResultReference:
        """Condense this RESULT into what may appear as TASK context."""
        return ResultReference(
            result_id=self.result_id,
            task_id=self.task_id,
            agent_id=self.agent_id,
            status=self.status,
            summary={
                "action_taken": dict(self.action_taken),
                "outcome": dict(self.outcome),
                "needs": [need.model_dump(mode="json") for need in self.needs],
                "offers": [offer.model_dump(mode="json") for offer in self.offers],
                "observations": [o.model_dump(mode="json") for o in self.observations],
            },
        )

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DirectorDecision(StrictProtocolModel):
    """One Director inference.

    Design s3.1 admits exactly two valid outputs: assign one TASK, or close the
    macro-cycle. Deliberative inferences that do neither are not permitted, so
    the schema cannot express one.
    """

    decision: Literal["assign", "close"]
    task: Task | None = None

    @model_validator(mode="after")
    def _decision_matches_payload(self) -> DirectorDecision:
        if self.decision == "assign" and self.task is None:
            raise ValueError("an 'assign' decision must carry a TASK")
        if self.decision == "close" and self.task is not None:
            raise ValueError("a 'close' decision must not carry a TASK")
        return self


class InvalidDirectorOutput(Exception):
    """Raised when a Director completion is not a valid decision."""
