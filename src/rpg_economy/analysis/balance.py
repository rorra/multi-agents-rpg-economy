"""Supply and demand of every good over a run, read from its log.

For each raw material: what expeditions brought in and what artisans used up.
For each product: what artisans made, what adventurers bought, what trips used
up, and what was left unsold. A healthy economy uses most of what comes in and
sells most of what is made; a good piling up, or one nobody buys, shows here
before it distorts prices. Reads a log only, rule-based or real.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rpg_economy.domain import Product, Resource
from rpg_economy.i18n import t


@dataclass(frozen=True)
class GoodBalance:
    good: str
    raw: bool
    #: Raw: gathered on expeditions. Product: made by artisans.
    supplied: int
    #: Raw: used up by artisans' recipes. Product: bought by adventurers.
    demanded: int
    #: Product: used up on trips (potions). Raw: 0.
    consumed: int
    traded: int
    #: Held at the end, by adventurers and by artisans (None: the log has no end).
    held_by_adventurers: int | None
    held_by_artisans: int | None

    @property
    def share(self) -> float | None:
        """Demanded over supplied: the part of what came in that found a use."""
        return self.demanded / self.supplied if self.supplied else None


@dataclass(frozen=True)
class RunBalance:
    log: str
    ticks: int
    expeditions: int
    goods: list[GoodBalance]


def balance_from_log(path: Path) -> RunBalance:
    """Tally the flows of every good in one run log."""
    supplied: Counter[str] = Counter()
    demanded: Counter[str] = Counter()
    consumed: Counter[str] = Counter()
    traded: Counter[str] = Counter()
    holdings: dict[str, dict] | None = None
    ticks = expeditions = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            kind, payload = event["type"], event["payload"]
            ticks = max(ticks, event.get("tick") or 0)
            if kind == "expedition_started":
                expeditions += 1
            elif kind == "expedition_resolved":
                for good, units in (payload.get("yielded") or {}).items():
                    supplied[good] += units
            elif kind == "production_completed":
                supplied[payload["product"]] += payload["quantity"]
                for good, units in payload["consumed"].items():
                    demanded[good] += units
            elif kind == "consumption":
                consumed[payload["product"]] += payload["quantity"]
            elif kind == "trade_executed":
                traded[payload["good"]] += payload["quantity"]
                if payload["good"] in PRODUCTS and payload["buyer_role"] == "adventurer":
                    demanded[payload["good"]] += payload["quantity"]
            elif kind == "run_end":
                holdings = payload.get("holdings")
    goods = []
    for good in [r.value for r in Resource] + [p.value for p in Product]:
        held = {"adventurer": 0, "artisan": 0}
        for account in (holdings or {}).values():
            held[account["role"]] = held.get(account["role"], 0) + int(
                account["inventory"].get(good, 0)
            )
        goods.append(
            GoodBalance(
                good=good,
                raw=good in RAW,
                supplied=supplied[good],
                demanded=demanded[good],
                consumed=consumed[good],
                traded=traded[good],
                held_by_adventurers=held["adventurer"] if holdings is not None else None,
                held_by_artisans=held["artisan"] if holdings is not None else None,
            )
        )
    return RunBalance(log=str(path), ticks=ticks, expeditions=expeditions, goods=goods)


RAW = frozenset(r.value for r in Resource)
PRODUCTS = frozenset(p.value for p in Product)


def _cell(value: int | None) -> str:
    return "-" if value is None else str(value)


def render_balance(balance: RunBalance) -> str:
    """The balance as two plain tables, in the reader's language."""
    lines = [
        t(
            "balance.header",
            log=balance.log,
            ticks=balance.ticks,
            expeditions=balance.expeditions,
        ),
        "",
        t("balance.raw_title"),
        t("balance.raw_columns"),
    ]
    for good in balance.goods:
        if not good.raw:
            continue
        share = "-" if good.share is None else f"{good.share:.0%}"
        lines.append(
            f"  {good.good:15s} {good.supplied:>8d} {good.demanded:>8d} {share:>6s} "
            f"{good.traded:>7d} {_cell(good.held_by_adventurers):>9s} "
            f"{_cell(good.held_by_artisans):>8s}"
        )
    lines += ["", t("balance.product_title"), t("balance.product_columns")]
    for good in balance.goods:
        if good.raw:
            continue
        share = "-" if good.share is None else f"{good.share:.0%}"
        lines.append(
            f"  {good.good:15s} {good.supplied:>8d} {good.demanded:>8d} {share:>6s} "
            f"{good.consumed:>8d} {_cell(good.held_by_adventurers):>9s} "
            f"{_cell(good.held_by_artisans):>8s}"
        )
    return "\n".join(lines)
