"""Token accounting, separated into buckets (design v1.0 s8.4, s15).

Experiment 2's budget unit is **tokens only** -- input plus output. Call count is
a secondary metric and never a budget unit. The design requires consumption be
separable into four buckets:

* ``agent`` -- economic agents' own prompts and completions;
* ``director`` -- Condition A's Director, overhead in Experiment 1 but charged to
  the same budget in Experiment 2;
* ``p2p_content`` -- peer-to-peer message content carried inside an agent prompt.
  Structurally absent in Condition A, where agents send no messages
  (design s3.1);
* ``bulletin`` -- Condition C's bulletin, counted **only when it is actually
  included in a prompt** (s3.3).

**How attribution works.** A provider reports one input-token total for a call,
not a breakdown by section. Splitting it therefore needs a local measure: each
prompt is assembled from labelled sections, each section is measured with the
same deterministic estimator, and the provider's authoritative total is
apportioned across sections in those proportions. Largest-remainder rounding
makes the parts sum to the total exactly, so no token is invented or lost. The
provider's number remains the ground truth for cost; the estimator only decides
how that cost is shared out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rpg_economy.domain import TokenBucket

#: Rough tokens-per-word, used only to apportion a call's input tokens between
#: sections. Never used as a cost figure.
_WORD_RE = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """A deterministic, offline size estimate for a piece of prompt text.

    Deliberately simple and dependency-free: it must give identical results on
    every machine and in every run, since bucket shares feed the logged
    accounting. It is a *ratio* instrument, not a price.
    """
    if not text:
        return 0
    return max(1, len(_WORD_RE.findall(text)))


def apportion(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Split ``total`` across ``weights``, summing exactly to ``total``.

    Largest-remainder apportionment. Rounding each share independently would
    lose or invent tokens, and the bucket totals must reconcile with the
    provider's figure exactly (IMPLEMENTATION.md section 6).
    """
    positive = {k: w for k, w in weights.items() if w > 0}
    if total <= 0 or not positive:
        return dict.fromkeys(weights, 0)

    denominator = sum(positive.values())
    exact = {k: total * w / denominator for k, w in positive.items()}
    floors = {k: int(v) for k, v in exact.items()}
    remainder = total - sum(floors.values())

    # Hand out the leftover to the largest fractional parts, ties by name so the
    # result never depends on dict ordering.
    ranked = sorted(positive, key=lambda k: (-(exact[k] - floors[k]), k))
    for key in ranked[:remainder]:
        floors[key] += 1

    return {k: floors.get(k, 0) for k in weights}


@dataclass
class TokenLedger:
    """Running token consumption for one run."""

    input_by_bucket: dict[TokenBucket, int] = field(
        default_factory=lambda: dict.fromkeys(TokenBucket, 0)
    )
    output_by_bucket: dict[TokenBucket, int] = field(
        default_factory=lambda: dict.fromkeys(TokenBucket, 0)
    )
    calls_by_bucket: dict[TokenBucket, int] = field(
        default_factory=lambda: dict.fromkeys(TokenBucket, 0)
    )

    #: Sum of provider-reported totals. This is the **contractual** figure the
    #: Experiment 2 budget consumes (design s8.4); the buckets above are
    #: analytical attribution of the same spend. They agree by construction,
    #: since apportionment is largest-remainder -- but where they could ever
    #: diverge, this one governs.
    authoritative_total: int = 0

    #: The subset of ``authoritative_total`` the account was actually billed for.
    #: Real and mock spend must be structurally unmixable, so mock figures are
    #: never reported as real money (IMPLEMENTATION.md section 4.7). Every
    #: real-spend total computes over this.
    billed_total: int = 0
    billed_calls: int = 0

    # -- recording ---------------------------------------------------------
    def record(
        self,
        *,
        input_by_bucket: dict[TokenBucket, int],
        output_tokens: int,
        output_bucket: TokenBucket,
        authoritative_total: int | None = None,
        billed: bool = False,
    ) -> None:
        """Record one LLM call.

        ``billed`` must come from the provider response, never from
        configuration: a config saying ``mock: false`` is a statement of intent,
        not evidence of billing. Only a parsed usage object counts.

        Output tokens belong wholly to whoever was speaking -- the agent or the
        Director. Only *input* is shared out, because only input carries
        embedded P2P content and bulletin text.

        ``authoritative_total`` is the provider's own figure for the call. It
        defaults to the recorded input plus output, which is what it equals in
        normal operation; it is passed explicitly where the provider's number
        must govern regardless of attribution (design s8.4).
        """
        recorded = sum(input_by_bucket.values()) + output_tokens
        charged = recorded if authoritative_total is None else authoritative_total
        self.authoritative_total += charged
        if billed:
            self.billed_total += charged
            self.billed_calls += 1
        for bucket, tokens in input_by_bucket.items():
            self.input_by_bucket[bucket] = self.input_by_bucket.get(bucket, 0) + tokens
        self.output_by_bucket[output_bucket] = (
            self.output_by_bucket.get(output_bucket, 0) + output_tokens
        )
        self.calls_by_bucket[output_bucket] = self.calls_by_bucket.get(output_bucket, 0) + 1

    # -- queries -----------------------------------------------------------
    def input_tokens(self, bucket: TokenBucket | None = None) -> int:
        if bucket is None:
            return sum(self.input_by_bucket.values())
        return self.input_by_bucket.get(bucket, 0)

    def output_tokens(self, bucket: TokenBucket | None = None) -> int:
        if bucket is None:
            return sum(self.output_by_bucket.values())
        return self.output_by_bucket.get(bucket, 0)

    def total(self, bucket: TokenBucket | None = None) -> int:
        """Input plus output -- the Experiment 2 budget unit."""
        return self.input_tokens(bucket) + self.output_tokens(bucket)

    @property
    def calls(self) -> int:
        """Total LLM calls. A secondary metric, never a budget unit."""
        return sum(self.calls_by_bucket.values())

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Language-neutral summary for the run log."""
        return {
            "input": {b.value: self.input_tokens(b) for b in TokenBucket},
            "output": {b.value: self.output_tokens(b) for b in TokenBucket},
            "calls": {b.value: self.calls_by_bucket.get(b, 0) for b in TokenBucket},
            "total": {b.value: self.total(b) for b in TokenBucket},
            "grand_total": self.total(),
            # The contractual figure. Budget enforcement reads this, never the
            # bucket sum above (design s8.4).
            "authoritative_total": self.authoritative_total,
            # Spend the account was actually billed for. Equals
            # authoritative_total for a fully real run and is zero for a mock
            # one; anything between means a run mixed them.
            "billed_total": self.billed_total,
            "billed_calls": self.billed_calls,
        }
