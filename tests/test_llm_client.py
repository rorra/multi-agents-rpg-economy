"""The LLM client interface and token accounting (design v1.0 s8.4, s15)."""

from __future__ import annotations

import ast

import pytest

from rpg_economy.domain import TokenBucket
from rpg_economy.llm import (
    LLMError,
    MockLLMClient,
    TokenLedger,
    always,
    apportion,
    build_client,
    estimate_tokens,
    scripted,
)
from tests.conftest import ROOT, make_config


# ---------------------------------------------------------------------------
# Offline by default
# ---------------------------------------------------------------------------
def test_mock_is_the_default_so_no_run_reaches_the_network() -> None:
    assert make_config().llm.mock is True


def test_build_client_returns_the_mock_when_configured() -> None:
    client = build_client(make_config().llm, always('{"action":"wait"}'))
    assert isinstance(client, MockLLMClient)
    assert client.model.startswith("mock:")


def test_the_mock_needs_a_responder() -> None:
    with pytest.raises(LLMError, match="needs a responder"):
        build_client(make_config().llm)


def test_importing_the_client_module_needs_no_provider_sdk() -> None:
    """Merely importing must not require the optional 'llm' extra."""
    import rpg_economy.llm.client as module

    assert module.OpenAIClient is not None


def test_only_the_client_module_loads_credentials() -> None:
    """IMPLEMENTATION.md section 3: dotenv is read in llm/client.py and nowhere else."""
    offenders = []
    for path in sorted((ROOT / "src/rpg_economy").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n.split(".")[0] == "dotenv" for n in names):
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == ["src/rpg_economy/llm/client.py"]


# ---------------------------------------------------------------------------
# The mock
# ---------------------------------------------------------------------------
def test_mock_reports_usage_and_counts_calls() -> None:
    client = MockLLMClient(always('{"action":"wait"}'))
    response = client.complete(system="sys", user="usr")

    assert response.output_tokens > 0
    assert response.input_tokens > 0
    assert response.total_tokens == response.input_tokens + response.output_tokens
    assert client.calls == 1


def test_mock_records_every_prompt() -> None:
    client = MockLLMClient(always("{}"))
    client.complete(system="a", user="b")
    client.complete(system="c", user="d")
    assert client.prompts == [("a", "b"), ("c", "d")]


def test_scripted_responder_returns_in_order() -> None:
    client = MockLLMClient(scripted(["one", "two"]))
    assert client.complete(system="", user="").text == "one"
    assert client.complete(system="", user="").text == "two"


def test_scripted_responder_can_be_exhausted() -> None:
    client = MockLLMClient(scripted(["only"]))
    client.complete(system="", user="")
    with pytest.raises(LLMError, match="exhausted"):
        client.complete(system="", user="")


def test_scripted_responder_can_loop() -> None:
    client = MockLLMClient(scripted(["a", "b"], loop=True))
    assert [client.complete(system="", user="").text for _ in range(4)] == ["a", "b", "a", "b"]


def test_the_mock_is_deterministic() -> None:
    def run() -> list[int]:
        client = MockLLMClient(always('{"action":"wait"}'))
        return [client.complete(system="s", user="u").output_tokens for _ in range(3)]

    assert run() == run()


# ---------------------------------------------------------------------------
# Token estimation and apportionment
# ---------------------------------------------------------------------------
def test_estimate_is_deterministic_and_monotone() -> None:
    assert estimate_tokens("buy iron") == estimate_tokens("buy iron")
    assert estimate_tokens("buy iron now please") > estimate_tokens("buy iron")
    assert estimate_tokens("") == 0


def test_apportion_sums_exactly_to_the_total() -> None:
    """Rounding each share alone would invent or lose tokens."""
    for total in (0, 1, 7, 13, 100, 9999):
        shares = apportion(total, {"a": 3, "b": 5, "c": 2})
        assert sum(shares.values()) == total


def test_apportion_respects_proportions() -> None:
    shares = apportion(1000, {"a": 9, "b": 1})
    assert shares["a"] == 900
    assert shares["b"] == 100


def test_apportion_ignores_absent_sections() -> None:
    shares = apportion(50, {"agent": 5, "bulletin": 0})
    assert shares["bulletin"] == 0
    assert shares["agent"] == 50


def test_apportion_is_order_independent() -> None:
    assert apportion(101, {"a": 1, "b": 1, "c": 1}) == apportion(101, {"c": 1, "b": 1, "a": 1})


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------
def test_buckets_sum_to_the_totals() -> None:
    """IMPLEMENTATION.md section 6: token buckets must reconcile with the totals."""
    ledger = TokenLedger()
    ledger.record(
        input_by_bucket={
            TokenBucket.AGENT: 100,
            TokenBucket.P2P_CONTENT: 20,
            TokenBucket.BULLETIN: 5,
        },
        output_tokens=30,
        output_bucket=TokenBucket.AGENT,
    )
    ledger.record(
        input_by_bucket={TokenBucket.DIRECTOR: 40},
        output_tokens=10,
        output_bucket=TokenBucket.DIRECTOR,
    )

    assert ledger.input_tokens() == 165
    assert ledger.output_tokens() == 40
    assert ledger.total() == 205
    assert sum(ledger.total(b) for b in TokenBucket) == ledger.total()


def test_the_budget_unit_is_input_plus_output() -> None:
    """Design v1.0 s8.4: tokens only, calls are a secondary metric."""
    ledger = TokenLedger()
    ledger.record(
        input_by_bucket={TokenBucket.AGENT: 7},
        output_tokens=3,
        output_bucket=TokenBucket.AGENT,
    )
    assert ledger.total(TokenBucket.AGENT) == 10
    assert ledger.calls == 1


def test_calls_are_counted_separately_from_tokens() -> None:
    ledger = TokenLedger()
    for _ in range(3):
        ledger.record(
            input_by_bucket={TokenBucket.AGENT: 1},
            output_tokens=1,
            output_bucket=TokenBucket.AGENT,
        )
    assert ledger.calls == 3
    assert ledger.total() == 6


def test_snapshot_is_language_neutral() -> None:
    ledger = TokenLedger()
    ledger.record(
        input_by_bucket={TokenBucket.AGENT: 5},
        output_tokens=2,
        output_bucket=TokenBucket.AGENT,
    )
    snapshot = ledger.snapshot()
    assert set(snapshot["input"]) == {b.value for b in TokenBucket}
    assert snapshot["grand_total"] == 7


# ---------------------------------------------------------------------------
# Provider failures (IMPLEMENTATION.md section 3)
# ---------------------------------------------------------------------------
class _StatusError(Exception):
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code
        self.body = body


class APITimeoutError(Exception):
    """Named like the SDK's, which is how the client recognises it."""


class _FlakyCompletions:
    def __init__(self, failures: list[Exception]) -> None:
        self._failures = list(failures)
        self.calls = 0

    def create(self, **request):
        from types import SimpleNamespace

        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"action":"wait"}'), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            model=request["model"],
        )


def _client_with(failures: list[Exception], **llm_overrides):
    """An OpenAIClient wired to a fake SDK, so no request leaves the machine."""
    from types import SimpleNamespace

    from rpg_economy.llm.client import OpenAIClient
    from tests.conftest import make_config

    config = make_config(llm={"mock": False, **llm_overrides}).llm
    completions = _FlakyCompletions(failures)
    client = object.__new__(OpenAIClient)
    client._config = config
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._sleep = lambda seconds: None
    import random

    client._jitter = random.Random(0)
    client._last, client._repeats = None, 0
    return client, completions


def test_transient_failures_are_retried_until_they_clear() -> None:
    client, completions = _client_with(
        [_StatusError(429), APITimeoutError("slow"), _StatusError(503)]
    )
    response = client.complete(system="s", user="u")
    assert completions.calls == 4
    assert response.billed and response.input_tokens == 100


def test_the_request_names_the_dated_snapshot() -> None:
    client, _ = _client_with([])
    assert client.complete(system="s", user="u").model == client._config.model_snapshot


@pytest.mark.invariant
@pytest.mark.parametrize(
    "failure",
    [
        _StatusError(401),
        _StatusError(400),
        _StatusError(403),
        _StatusError(429, {"code": "insufficient_quota"}),
    ],
)
def test_a_refused_request_stops_at_once(failure: Exception) -> None:
    from rpg_economy.llm.client import ProviderError

    client, completions = _client_with([failure])
    with pytest.raises(ProviderError) as caught:
        client.complete(system="s", user="u")
    assert completions.calls == 1
    assert not caught.value.transient


def test_a_failure_that_outlasts_its_retries_stops_the_run() -> None:
    from rpg_economy.llm.client import ProviderError

    client, completions = _client_with([_StatusError(503)] * 10, max_transient_retries=3)
    with pytest.raises(ProviderError) as caught:
        client.complete(system="s", user="u")
    assert completions.calls == 4
    assert caught.value.transient


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 520, 529])
def test_rate_limits_and_server_failures_are_transient(status: int) -> None:
    from rpg_economy.llm.client import is_transient

    assert is_transient(_StatusError(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_transient(status: int) -> None:
    from rpg_economy.llm.client import is_transient

    assert not is_transient(_StatusError(status))


def test_an_exhausted_quota_is_not_transient_even_as_a_429() -> None:
    from rpg_economy.llm.client import is_transient

    assert not is_transient(_StatusError(429, {"code": "insufficient_quota"}))


def test_an_unlisted_server_status_is_retried_until_it_clears() -> None:
    client, completions = _client_with([_StatusError(529), _StatusError(520)])
    response = client.complete(system="s", user="u")
    assert completions.calls == 3
    assert response.billed


def test_a_rate_limit_waits_as_long_as_the_provider_asks() -> None:
    from types import SimpleNamespace

    limited = _StatusError(429)
    limited.response = SimpleNamespace(headers={"retry-after": "7"})
    client, _ = _client_with([limited])
    waits: list[float] = []
    client._sleep = waits.append
    client.complete(system="s", user="u")
    assert waits == [7.0]


def test_a_retry_always_waits_at_least_half_its_delay() -> None:
    client, _ = _client_with([_StatusError(503)] * 3)
    waits: list[float] = []
    client._sleep = waits.append
    client.complete(system="s", user="u")
    base = float(client._config.retry_base_delay_s)
    for attempt, wait in enumerate(waits):
        assert wait >= base * 2**attempt / 2


def test_an_answer_with_no_choices_is_retried_then_transient() -> None:
    from types import SimpleNamespace

    from rpg_economy.llm.client import ProviderError

    client, completions = _client_with([])
    calls = []

    def empty(**request):
        calls.append(1)
        return SimpleNamespace(choices=[], usage=None, model="m")

    completions.create = empty
    with pytest.raises(ProviderError) as caught:
        client.complete(system="s", user="u")
    assert caught.value.transient
    assert len(calls) == client._config.max_transient_retries + 1


@pytest.mark.parametrize(
    ("headers", "seconds"),
    [
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after": "7"}, 7.0),
        ({"x-ratelimit-reset-requests": "1m30s", "x-ratelimit-reset-tokens": "120ms"}, 90.0),
        ({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}, 0.0),
    ],
)
def test_the_provider_wait_is_read_from_its_headers(headers: dict, seconds: float) -> None:
    from types import SimpleNamespace

    from rpg_economy.llm.client import retry_after

    exc = Exception("limited")
    exc.response = SimpleNamespace(headers=headers)
    assert retry_after(exc) == pytest.approx(seconds)


def test_a_resend_of_the_same_prompt_uses_the_next_seed() -> None:
    """A retry after an invalid answer must not replay the same draw."""
    client, completions = _client_with([])
    seeds: list[int] = []
    create = completions.create

    def recording(**request):
        seeds.append(request["seed"])
        return create(**request)

    completions.create = recording
    client.complete(system="s", user="u")
    client.complete(system="s", user="u")
    client.complete(system="s", user="other")
    base = int(client._config.seed)
    assert seeds == [base, base + 1, base]
    assert client.complete(system="s", user="other").seed == base + 1
