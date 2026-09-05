"""Event vocabulary and envelope for the JSONL run log.

Design v1.0 section 15 requires that a run be **fully reconstructable from its
log**. The event types below enumerate everything the design demands be
recorded; later phases fill in their payloads.

Two rules govern this module:

1. **No wall-clock data in events.** Time is measured in ticks and macro-cycles.
   A timestamp would make two runs of the same seed differ byte for byte and
   destroy the determinism test. Wall-clock and host details live in the run
   metadata's ``environment`` block, which the determinism digest excludes.
2. **Language-neutral values only.** Payloads carry enum values and numbers;
   translation happens at render time (IMPLEMENTATION.md section 2).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """Everything design v1.0 section 15 requires to be logged."""

    # --- run lifecycle -----------------------------------------------------
    RUN_METADATA = "run_metadata"
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    TICK = "tick"
    RUN_END = "run_end"
    #: The run was stopped mid-way at the operator's request. Never followed by
    #: run_end, so the log reads as unfinished and a resume re-runs it.
    RUN_ABORTED = "run_aborted"

    # --- activation machinery (B/C) ---------------------------------------
    TRIGGER_FIRED = "trigger_fired"
    ACTIVATION_REQUESTED = "activation_requested"
    ACTIVATION_SERVED = "activation_served"
    ACTIVATION_DEFERRED = "activation_deferred"
    #: A standing request whose triggers all stopped holding before it was served.
    ACTIVATION_REQUEST_RESOLVED = "activation_request_resolved"

    # --- agent decision cycle (all conditions) -----------------------------
    OBSERVATION_DELIVERED = "observation_delivered"
    AGENT_OUTPUT = "agent_output"
    AGENT_OUTPUT_INVALID = "agent_output_invalid"
    ACTION_EXECUTED = "action_executed"
    ACTION_REJECTED = "action_rejected"
    ACTIVATION_WASTED = "activation_wasted"

    #: A system prompt, in full, logged once per run the first time it is sent.
    #: Calls name it by ``system_sha`` and carry their own user prompt, so the
    #: exact text of every call is in the log without repeating the fixed part.
    PROMPT_SYSTEM = "prompt_system"

    # --- Director (A only) -------------------------------------------------
    DIRECTOR_CALL = "director_call"
    #: The cycle's global view, in full, as the Director reads it (once per cycle).
    DIRECTOR_VIEW = "director_view"
    DIRECTOR_TASK_ASSIGNED = "director_task_assigned"
    #: A TASK whose assigning call spent the last of the budget: never carried out.
    DIRECTOR_TASK_DROPPED = "director_task_dropped"
    DIRECTOR_CYCLE_CLOSED = "director_cycle_closed"
    DIRECTOR_ERROR = "director_error"
    RESULT_RETURNED = "result_returned"

    # --- messaging ---------------------------------------------------------
    MESSAGE_SENT = "message_sent"
    #: Inbox content carried by one LLM call. Logged per attempt, so a retry
    #: repeats it with that call's tokens: summing ``tokens`` is right, but
    #: counting deliveries must de-duplicate by the payload's ``activation``
    #: (tick, agent_id, seq), ignoring ``attempt``.
    MESSAGE_DELIVERED = "message_delivered"

    # --- bulletin (C only) -------------------------------------------------
    BULLETIN_PUBLISHED = "bulletin_published"
    #: The bulletin carried by one LLM call; per attempt, as MESSAGE_DELIVERED.
    #: Count inclusions de-duplicated by ``activation``; sum ``tokens`` as is.
    BULLETIN_INCLUDED_IN_PROMPT = "bulletin_included_in_prompt"

    # --- market ------------------------------------------------------------
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_EXPIRED = "order_expired"
    ORDER_CANCELLED = "order_cancelled"
    TRADE_EXECUTED = "trade_executed"

    # --- world -------------------------------------------------------------
    PRODUCTION_COMPLETED = "production_completed"
    PRODUCTION_REJECTED = "production_rejected"
    EXPEDITION_STARTED = "expedition_started"
    EXPEDITION_RESOLVED = "expedition_resolved"
    EQUIPMENT_EQUIPPED = "equipment_equipped"
    EQUIPMENT_BROKEN = "equipment_broken"
    DURABILITY_DECAYED = "durability_decayed"
    CONSUMPTION = "consumption"
    STORAGE_PAID = "storage_paid"
    TAX_PAID = "tax_paid"
    RESOURCES_SPAWNED = "resources_spawned"
    SHOCK_APPLIED = "shock_applied"
    SHOCK_REVERTED = "shock_reverted"
    GOODS_CONFISCATED = "goods_confiscated"

    # --- cost accounting ---------------------------------------------------
    LLM_CALL = "llm_call"
    TOKENS_CONSUMED = "tokens_consumed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Event(BaseModel):
    """One immutable line of the run log."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seq: int = Field(ge=0, description="Monotonic index within the run. Gapless.")
    tick: int = Field(ge=0, description="Engine tick at which the event occurred.")
    cycle: int = Field(ge=0, description="Macro-cycle index.")
    type: EventType
    agent_id: str | None = Field(default=None, description="Subject agent, when the event has one.")
    payload: dict[str, Any] = Field(default_factory=dict)


def prompt_sha(text: str) -> str:
    """Short digest naming a system prompt in call events."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
