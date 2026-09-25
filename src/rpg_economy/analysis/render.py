"""Text rendering of the economic sanity check.

Every string resolves through ``i18n.t``, so the same computed result renders in
English or Spanish without recomputation (IMPLEMENTATION.md section 2).
"""

from __future__ import annotations

from decimal import Decimal

from rpg_economy.analysis.economy_check import EconomyCheck, PhaseStats
from rpg_economy.config.models import SimulationConfig
from rpg_economy.i18n import format_number, t

CENTS = Decimal("0.01")


def _money(value: Decimal | None) -> str:
    """Two decimals with the active locale's separators."""
    return "-" if value is None else format_number(value.quantize(CENTS), 2)


def _count(value: int | None) -> str:
    """A count with the active locale's thousands separator."""
    return "-" if value is None else format_number(value)


def _row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [_row(headers, widths), _row(["-" * w for w in widths], widths)]
    lines.extend(_row(row, widths) for row in rows)
    return lines


def render(check: EconomyCheck, config: SimulationConfig) -> str:
    """Render the diagnostic as plain text in the active locale."""
    lines: list[str] = [
        t("check.title"),
        t(
            "check.run_line",
            # The check runs rule-based agents under the decentralized
            # activation regime whatever ``config.run.condition`` says, so the
            # header names that regime rather than the configured condition.
            condition=t("check.regime.rule_based"),
            scenario=t(f"scenario.{config.run.scenario.value}"),
            experiment=t(f"experiment.{config.run.experiment.value}"),
            seed=config.run.seed,
            ticks=config.time.horizon_ticks,
            cycles=config.time.total_cycles,
        ),
        "",
    ]

    def phase_label(stats: PhaseStats) -> str:
        return f"{t(f'phase.{stats.phase}')} [{stats.first_tick}-{stats.last_tick})"

    if check.target_good is None:
        # liquidity_loss: the pre-registered target is total traded value
        # (design s11.3), so that is what the first section shows.
        lines.append(t("check.section.traded_value"))
        lines += _table(
            [t("check.col.phase"), t("check.col.trades"), t("check.col.value")],
            [[phase_label(p), _count(p.trades), _money(p.traded_value)] for p in check.phases],
        )
    else:
        lines += _price_section(check, phase_label)

    lines += ["", t("check.section.activity")]
    lines += _table(
        [
            t("check.col.phase"),
            t("check.col.ticks"),
            t("check.col.trades"),
            t("check.col.per_tick"),
            t("check.col.value"),
        ],
        [
            [
                phase_label(p),
                _count(p.ticks),
                _count(p.trades),
                format_number(p.trades_per_tick.quantize(CENTS), 2),
                _money(p.traded_value),
            ]
            for p in check.phases
        ],
    )

    if check.target_good is not None:
        lines += ["", t("check.section.stockouts")]
    for stockout in check.stockouts:
        if stockout.total_ticks == 0:
            lines.append(t("check.stockout.none"))
            continue
        # Episodes and the longest episode span phases, so they are whole-run
        # figures: they get their own row rather than sitting on the base row.
        lines += _table(
            [
                t("check.col.phase"),
                t("check.col.ticks"),
                t("check.col.episodes"),
                t("check.col.longest"),
            ],
            [
                *(
                    [t(f"phase.{phase}"), _count(stockout.by_phase.get(phase, 0)), "", ""]
                    for phase in ("base", "shock", "recovery")
                ),
                [
                    t("check.row.whole_run"),
                    _count(stockout.total_ticks),
                    _count(stockout.episodes),
                    _count(stockout.longest),
                ],
            ],
        )

    lines += _holdings_and_health(check)
    return "\n".join(lines)


def _good_label(good: str) -> str:
    resources = {"iron", "wood", "leather", "herbs", "arcane_crystal"}
    return t(f"resource.{good}") if good in resources else t(f"product.{good}")


def _price_section(check: EconomyCheck, phase_label) -> list[str]:
    lines = [t("check.section.prices", good=_good_label(check.target_good or ""))]
    lines += _table(
        [
            t("check.col.phase"),
            t("check.col.mean_price"),
            t("check.col.min_price"),
            t("check.col.max_price"),
            t("check.col.trades"),
            t("check.col.units"),
            t("check.col.gathered"),
        ],
        [
            [
                phase_label(p),
                _money(p.mean_target_price),
                _money(p.min_target_price),
                _money(p.max_target_price),
                _count(p.target_trades),
                _count(p.target_units),
                _count(check.target_gathered.get(p.phase, 0)),
            ]
            for p in check.phases
        ],
    )
    return lines


def _holdings_and_health(check: EconomyCheck) -> list[str]:
    lines: list[str] = []
    lines += ["", t("check.section.holdings")]
    lines += _table(
        [
            t("check.col.role"),
            t("check.col.agents"),
            t("check.col.gold_start"),
            t("check.col.gold_end"),
            t("check.col.gold_change"),
            t("check.col.units_start"),
            t("check.col.units_end"),
        ],
        [
            [
                t(f"role.{row.role}"),
                _count(row.agents),
                _money(row.gold_start),
                _money(row.gold_end),
                ("+" if row.gold_change >= 0 else "") + format_number(row.gold_change, 2),
                _count(row.units_start),
                _count(row.units_end),
            ]
            for row in check.roles
        ],
    )

    lines += ["", t("check.section.health")]
    balanced = (
        check.total_gold_start
        + check.gold_looted
        + check.gold_confiscation_paid
        - check.gold_taxed
        - check.gold_storage
        == check.total_gold_end
    )
    lines.append(
        t(
            "check.health.gold_balanced" if balanced else "check.health.gold_unbalanced",
            start=_money(check.total_gold_start),
            loot=_money(check.gold_looted),
            confiscated=_money(check.gold_confiscation_paid),
            taxes=_money(check.gold_taxed),
            storage=_money(check.gold_storage),
            end=_money(check.total_gold_end),
        )
    )
    lines.append(
        t("check.health.reconciles" if check.gold_reconciles else "check.health.not_reconciles")
    )
    lines.append(
        t("check.health.no_bankruptcies")
        if not check.bankrupt_agents
        else t("check.health.bankruptcies", agents=", ".join(check.bankrupt_agents))
    )
    lines.append(
        t(
            "check.health.concentration",
            share=format_number((check.richest_share * 100).quantize(CENTS), 2),
        )
    )
    lines.append(
        t(
            "check.health.activity",
            expeditions=_count(check.expeditions),
            production=_count(check.production),
        )
    )
    return lines
