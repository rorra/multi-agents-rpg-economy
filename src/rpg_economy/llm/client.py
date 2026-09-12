"""The single isolated LLM client interface (IMPLEMENTATION.md section 3).

This is the **only** module permitted to touch the network or read credentials.
The engine has no dependency on it, and every part of the pipeline runs fully
offline against ``MockLLMClient`` -- including complete trio runs and the whole
analysis path.

The interface is provider-agnostic; OpenAI is this project's implementation. The
model name and inference parameters come from configuration and are recorded in
run metadata, never written as literals here (design v1.0 s15).
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rpg_economy.config.models import LLMConfig
from rpg_economy.llm.tokens import estimate_tokens


class LLMError(Exception):
    """Raised when a completion cannot be obtained."""


def _duration(text: str) -> float | None:
    """Seconds in a header value: "7", "250ms", "1.5s", "1m30s" or "2h"."""
    text = text.strip().lower()
    try:
        return float(text)
    except ValueError:
        pass
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", text)
    if not parts or "".join(n + u for n, u in parts) != text:
        return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(n) * scale[u] for n, u in parts)


def retry_after(exc: BaseException) -> float:
    """Seconds the provider asked to wait before retrying, or 0 when it did not say.

    ``retry-after-ms`` and ``retry-after`` first; otherwise the longer of the
    request and token rate-limit resets. An HTTP-date ``retry-after`` is
    ignored: the backoff floor still applies.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    get = headers.get if hasattr(headers, "get") else (lambda name: None)
    ms = get("retry-after-ms")
    if ms is not None:
        value = _duration(str(ms))
        if value is not None:
            return value / 1000
    after = get("retry-after")
    if after is not None:
        value = _duration(str(after))
        if value is not None:
            return value
    resets = [
        _duration(str(get(name)))
        for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")
        if get(name) is not None
    ]
    return max((v for v in resets if v is not None), default=0.0)


class EmptyCompletion(Exception):
    """The provider answered with no choices: a transient failure, retried."""


class ProviderError(LLMError):
    """The provider cannot answer, so the run must stop.

    Distinct from an invalid completion: a WAIT caused by an expired key, an
    exhausted quota or an outage is not a decision of the agent, and letting it
    count as one would produce a run that looks complete but measures the
    outage. Callers re-raise it instead of treating it as invalid output.

    ``transient`` says whether the failure was of a kind that may clear by
    itself (a rate limit or an outage that outlasted the retries), which is
    what a campaign may wait out before re-running the trio.
    """

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


#: Status codes worth retrying: rate limits and server-side failures.
#: Retried besides every 5xx: the provider's own capacity signals (520, 529
#: and the like) are server-side too, and waiting them out is the right call.
TRANSIENT_STATUS = frozenset({408, 409, 429})


def is_transient(exc: BaseException) -> bool:
    """Whether a provider exception is worth retrying.

    Classified by status code and exception type rather than by message text,
    so a file path or token count that happens to contain "500" never counts.
    An exhausted quota arrives as a 429 but will not recover by waiting.
    """
    body = getattr(exc, "body", None)
    code = (body or {}).get("code") if isinstance(body, dict) else getattr(exc, "code", None)
    if code == "insufficient_quota":
        return False
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in TRANSIENT_STATUS or status >= 500
    name = type(exc).__name__
    return "Timeout" in name or "Connection" in name


@dataclass(frozen=True)
class LLMResponse:
    """One completion, with the token usage that will be charged for it."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str | None = None
    #: Hidden reasoning tokens (billed as output) and input tokens served from
    #: the provider's cache (billed at a discount). Recorded, not charged apart:
    #: output_tokens already includes the reasoning.
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    #: The provider seed the request carried, logged so a run can be replayed
    #: call for call (a retry moves it on by one).
    seed: int | None = None

    #: True only when these figures came from a provider usage object that the
    #: account was actually billed for. A mock completion is never billed, and
    #: neither is a real response that arrived without usage -- in that case the
    #: numbers are our own estimate, and calling them spend would be a guess
    #: presented as a measurement (IMPLEMENTATION.md section 4.7).
    billed: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(Protocol):
    """What the rest of the system needs from a language model."""

    @property
    def model(self) -> str: ...

    def complete(self, *, system: str, user: str) -> LLMResponse:
        """Return one completion for a system/user prompt pair."""
        ...


# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------
#: A responder turns a (system, user) prompt pair into raw completion text.
Responder = Callable[[str, str], str]


class MockLLMClient:
    """A deterministic offline client.

    It exercises the *entire* pipeline -- prompt assembly, the call boundary,
    parsing, validation, the retry policy and token accounting -- while what it
    decides is supplied by a plain callable. That separation is what lets an
    offline run be economically meaningful without any of it being special-cased:
    the code path under test is the same one a real provider drives.
    """

    def __init__(self, responder: Responder, *, model: str = "mock") -> None:
        self._responder = responder
        self._model = model
        self._calls = 0
        self._prompts: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def prompts(self) -> list[tuple[str, str]]:
        """Every (system, user) pair seen. Used by tests to inspect prompts."""
        return list(self._prompts)

    def complete(self, *, system: str, user: str) -> LLMResponse:
        self._calls += 1
        self._prompts.append((system, user))
        text = self._responder(system, user)
        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(system) + estimate_tokens(user),
            output_tokens=estimate_tokens(text),
            model=self._model,
            finish_reason="stop",
        )


def scripted(responses: Sequence[str], *, loop: bool = False) -> Responder:
    """A responder that returns a fixed sequence of completions."""
    remaining = list(responses)
    index = 0

    def respond(system: str, user: str) -> str:
        nonlocal index
        if not remaining:
            raise LLMError("scripted responder has no responses configured")
        if index >= len(remaining):
            if not loop:
                raise LLMError("scripted responder exhausted")
            index = 0
        text = remaining[index]
        index += 1
        return text

    return respond


def always(text: str) -> Responder:
    """A responder that always returns the same completion."""
    return lambda system, user: text


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class OpenAIClient:
    """The provider implementation.

    Imports the SDK and reads credentials lazily, inside ``__init__``, so that
    merely importing this module never requires the optional dependency or an
    ``.env`` file. Offline runs never construct one.
    """

    def __init__(self, config: LLMConfig, *, api_key: str | None = None) -> None:
        try:
            from dotenv import load_dotenv
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover -- offline installs
            raise LLMError(
                "the OpenAI provider needs the 'llm' extra: poetry install --extras llm"
            ) from exc

        # Credentials are loaded here and nowhere else (IMPLEMENTATION.md section 3).
        load_dotenv()
        self._config = config
        # Retries are done here, where failures are classified, not inside the
        # SDK; and the timeout is explicit so one hung call cannot stall a run.
        options: dict[str, Any] = {"timeout": config.request_timeout_s, "max_retries": 0}
        if api_key:
            options["api_key"] = api_key
        self._client = OpenAI(**options)
        self._sleep = time.sleep
        self._jitter = random.Random()
        #: The last request and how many times in a row it was sent: a retry
        #: after an invalid answer resends the same prompt, and with the same
        #: seed a model that honours it would answer the same way again.
        self._last: tuple[str, str] | None = None
        self._repeats = 0

    @property
    def model(self) -> str:
        return self._config.model

    def _seed_for(self, system: str, user: str) -> int | None:
        """The configured seed, moved on by one for each identical resend.

        The same rule in every condition, so a retry is a fresh draw everywhere.
        """
        key = (system, user)
        self._repeats = self._repeats + 1 if key == self._last else 0
        self._last = key
        if self._config.seed is None:
            return None
        return int(self._config.seed) + self._repeats

    def complete(self, *, system: str, user: str) -> LLMResponse:
        # The dated snapshot, when known: the alias is a promise to return
        # whatever is current, which is not a fixed model (design s15).
        model = self._config.model_snapshot or self._config.model
        # max_completion_tokens, not max_tokens: the gpt-5 family rejects the
        # older name.
        request: dict[str, Any] = {
            "model": model,
            "temperature": float(self._config.temperature),
            "max_completion_tokens": self._config.max_output_tokens,
            "seed": self._seed_for(system, user),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._config.reasoning_effort is not None:
            # Without this a reasoning model spends the whole completion budget
            # thinking and returns empty content (design s13).
            request["reasoning_effort"] = self._config.reasoning_effort

        retries = self._config.max_transient_retries
        for attempt in range(retries + 1):
            try:
                response = self._client.chat.completions.create(**request)
                if not response.choices:
                    # Nothing in it: the provider failing, not the model deciding.
                    raise EmptyCompletion("provider returned no choices")
                break
            except Exception as exc:
                if not (isinstance(exc, EmptyCompletion) or is_transient(exc)):
                    raise ProviderError(f"provider refused the request: {exc}") from exc
                if attempt == retries:
                    raise ProviderError(
                        f"provider still failing after {retries} retries: {exc}",
                        transient=True,
                    ) from exc
                delay = min(
                    self._config.retry_max_delay_s,
                    self._config.retry_base_delay_s * 2**attempt,
                )
                # Half the delay at least, jitter on the rest, so parallel
                # workers do not return in lockstep and never retry at once; a
                # provider that says when to come back is waited for.
                wait = delay / 2 + self._jitter.uniform(0, delay / 2)
                self._sleep(min(self._config.retry_max_delay_s, max(wait, retry_after(exc))))

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            # The provider's figures are authoritative for cost; the local
            # estimator only ever decides how a call is split between buckets.
            input_tokens=usage.prompt_tokens if usage else estimate_tokens(system + user),
            # Without usage the output is estimated, never counted as free: a
            # zero here would let the budget undercharge.
            output_tokens=(
                usage.completion_tokens if usage else estimate_tokens(choice.message.content or "")
            ),
            model=response.model,
            finish_reason=choice.finish_reason,
            reasoning_tokens=_detail(usage, "completion_tokens_details", "reasoning_tokens"),
            cached_tokens=_detail(usage, "prompt_tokens_details", "cached_tokens"),
            billed=usage is not None,
            seed=request["seed"],
        )


def _detail(usage: object, group: str, field: str) -> int:
    """A usage sub-field, or 0 where the provider does not report it."""
    value = getattr(getattr(usage, group, None), field, None)
    return int(value) if isinstance(value, int) else 0


def build_client(config: LLMConfig, responder: Responder | None = None) -> LLMClient:
    """Construct the client the configuration asks for.

    ``llm.mock`` defaults to true, so no run reaches the network by accident.
    """
    if config.mock:
        if responder is None:
            raise LLMError("the mock client needs a responder")
        return MockLLMClient(responder, model=f"mock:{config.model}")
    return OpenAIClient(config)
