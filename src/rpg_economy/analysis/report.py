"""Bilingual report generation (design v1.0 s11; IMPLEMENTATION.md section 2).

Both languages render from one results file. Nothing about a language is
stored: the file holds enum values and numbers, and every string is applied here
through ``i18n.t``, so ``make report LOCALE=es`` and ``LOCALE=en`` describe the
same numbers.
"""

from __future__ import annotations

import statistics
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from rpg_economy.analysis.results import Results
from rpg_economy.i18n import t
from rpg_economy.i18n.catalog import get_catalog
from rpg_economy.i18n.numbers import SEPARATORS, format_number

CONDITION_ORDER = ("A", "B", "C")

_RESOURCES = {"iron", "wood", "leather", "herbs", "arcane_crystal"}


def good_label(good: str | None) -> str:
    """A good's name, or the aggregate-activity label when there is no good."""
    if good is None:
        return t("metric.volume")
    return t(f"resource.{good}") if good in _RESOURCES else t(f"product.{good}")


def _fmt(value: float | None, places: int = 3) -> str:
    """A number in the locale, with at most ``places`` decimals, trailing zeros cut."""
    if value is None or value != value:  # missing or NaN
        return t("report.value.missing")
    if abs(value) >= 1000:
        return format_number(value, 0)
    text = format_number(value, places)
    if places > 0:
        point = SEPARATORS[get_catalog().locale][1]
        text = text.rstrip("0").rstrip(point)
    if text in ("-0", ""):  # a tiny negative rounds to zero, not "minus zero"
        text = "0"
    return text


def _p(value: float | None) -> str:
    if value is None or value != value:  # missing or NaN
        return t("report.value.missing")
    return ("<" + format_number(0.001, 3)) if value < 0.001 else format_number(value, 3)


def _percent(value: float | None) -> str:
    """A share as a percentage with one decimal, rounded half up, in the locale.

    Rounded in decimal from the value's shortest representation, so 12.5% is
    "12.5%" (never "12%" by float half-to-even), and always one decimal, so a
    column of shares lines up.
    """
    if value is None or value != value:
        return t("report.value.missing")
    percent = (Decimal(repr(float(value))) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if percent == 0:
        percent = abs(percent)  # never "-0.0%"
    return format_number(percent, 1) + "%"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    line = lambda cells: "  ".join(  # noqa: E731
        cell.ljust(width) for cell, width in zip(cells, widths, strict=True)
    ).rstrip()
    return [line(headers), line(["-" * w for w in widths]), *(line(row) for row in rows)]


def metric_label(metric: str) -> str:
    return t(f"report.metric.{metric}")


def contrast_label(payload: dict[str, Any]) -> str:
    """Render a contrast's two conditions, e.g. "Centralized / Decentralized"."""
    left = t("condition." + payload["left"])
    right = t("condition." + payload["right"])
    return f"{left} / {right}"


def _median_table(results: Results, metrics: list[str]) -> list[str]:
    headers = [
        t("report.col.metric"),
        *(t(f"condition.{c}") for c in results.conditions),
        t("report.col.n"),
    ]
    rows = []
    suppressed_any = False
    for metric in metrics:
        table = results.by_condition(metric)
        if not any(table.values()):
            continue
        # n is the pairable count: seeds carrying a usable value in *every*
        # condition. The medians are taken over those same seeds, so each
        # column describes the runs n counts (design s11.3); per-condition
        # measurability is reported separately.
        pairable = set.intersection(*(set(table.get(c, {})) for c in results.conditions))
        if metric in results.suppressed_contrasts:
            # Measurability differs by condition, so the pairable seeds are the
            # ones the treatment let trade. A median over them would print the
            # very comparison the suppressed contrast withholds (design s11.3).
            suppressed_any = True
            rows.append(
                [
                    metric_label(metric),
                    *(t("report.value.suppressed") for _ in results.conditions),
                    _fmt(len(pairable), 0),
                ]
            )
            continue
        rows.append(
            [
                metric_label(metric),
                *(
                    _fmt(statistics.median(table[c][seed] for seed in pairable))
                    if pairable
                    else t("report.value.missing")
                    for c in results.conditions
                ),
                _fmt(len(pairable), 0),
            ]
        )
    lines = _table(headers, rows)
    if suppressed_any:
        lines.append(t("report.note.median_suppressed"))
    return lines


def _suppression_note(suppressed: dict[str, Any]) -> list[str]:
    pairs = ", ".join(contrast_label(pair) for pair in suppressed.get("pairs", []))
    return [
        t("report.note.contrast_suppressed_pairs", pairs=pairs),
        (
            t(
                "report.note.contrast_suppressed",
                companion=metric_label(suppressed["companion_metric"]),
            )
            if suppressed.get("companion_metric")
            else t("report.note.contrast_suppressed_no_companion")
        ),
    ]


def _recovery_tests(survival: dict[str, Any]) -> list[str]:
    """The paired binary and stratified log-rank tables of one recovery analysis.

    The same two tests for the primary recovery, the common-horizon recovery and
    the propagated product's: each was computed, so each is reported.
    """
    lines: list[str] = []
    suppressed = survival.get("suppressed_pairs") or []
    kept = [pair for pair in suppressed if pair.get("in_family", True)]
    left_out = [pair for pair in suppressed if not pair.get("in_family", True)]
    if kept:
        pairs = ", ".join(contrast_label(pair) for pair in kept)
        lines += ["", t("report.note.recovery_suppressed", pairs=pairs)]
    if left_out:
        pairs = ", ".join(contrast_label(pair) for pair in left_out)
        lines += ["", t("report.note.recovery_suppressed_no_pairs", pairs=pairs)]
    if survival.get("no_pairs"):
        pairs = ", ".join(contrast_label(pair) for pair in survival["no_pairs"])
        lines += ["", t("report.note.recovery_no_pairs", pairs=pairs)]
    if any(survival.get("unmeasurable", {}).values()) and survival.get("measurability"):
        lines += ["", t("report.test.paired_binary_recovery_measurable")]
        lines += _table(
            [
                t("report.col.contrast"),
                t("report.col.discordant"),
                t("report.col.p"),
                t("report.col.p_holm"),
            ],
            [
                [
                    contrast_label(r),
                    _fmt(r["discordant_pairs"], 0),
                    _p(r["p_value"]),
                    _p(r.get("p_value_holm")),
                ]
                for r in survival["measurability"]
            ],
        )
    if survival.get("paired_binary"):
        lines += ["", t("report.test.paired_binary")]
        lines += _table(
            [
                t("report.col.contrast"),
                t("report.col.discordant"),
                t("report.col.p"),
                t("report.col.p_holm"),
            ],
            [
                [
                    contrast_label(r),
                    _fmt(r["discordant_pairs"], 0),
                    _p(r["p_value"]),
                    _p(r.get("p_value_holm")),
                ]
                for r in survival["paired_binary"]
            ],
        )
    if survival.get("stratified_logrank"):
        lines += ["", t("report.test.logrank")]
        lines += _table(
            [t("report.col.contrast"), "z", t("report.col.p"), t("report.col.p_holm")],
            [
                [
                    contrast_label(r),
                    _fmt(r["statistic"], 2),
                    _p(r["p_value"]),
                    _p(r.get("p_value_holm")),
                ]
                for r in survival["stratified_logrank"]
            ],
        )
    return lines


def _recovery_table(results: Results, survival: dict[str, Any]) -> list[str]:
    # A condition in a withheld pair shows no median: over its measurable runs
    # only, it would show the comparison the withholding keeps back.
    withheld = {
        condition
        for pair in survival.get("suppressed_pairs") or []
        for condition in (pair["left"], pair["right"])
    }
    rows = []
    for condition in results.conditions:
        curve = survival["kaplan_meier"].get(condition, {})
        rows.append(
            [
                t(f"condition.{condition}"),
                f"{_fmt(curve.get('events', 0), 0)}/{_fmt(curve.get('n', 0), 0)}",
                _percent(survival["censoring_rate"].get(condition)),
                _fmt(survival.get("unmeasurable", {}).get(condition, 0), 0),
                # The Kaplan-Meier median, which counts censored runs; missing
                # when the curve never falls to one half.
                (
                    t("report.value.suppressed")
                    if condition in withheld
                    else _fmt(curve.get("median"), 1)
                ),
            ]
        )
    return _table(
        [
            t("report.col.condition"),
            t("report.col.recovered"),
            t("report.col.censored"),
            t("report.col.unmeasurable"),
            t("report.col.km_median"),
        ],
        rows,
    )


def _incomplete_trios_note(results: Results) -> list[str]:
    """One line naming the seeds dropped for an incomplete trio (design s11.3)."""
    if not results.incomplete_trios:
        return []
    seeds = ", ".join(str(seed) for seed in sorted(results.incomplete_trios))
    return [
        t("report.note.incomplete_trios", seeds=seeds, count=_fmt(len(results.incomplete_trios), 0))
    ]


#: The design's letter for each metric parameter (s11.1, s11.2), in report order.
METRIC_PARAMETER_LETTERS = {
    "volatility_window_w": "W",
    "recovery_band_pct_x": "X",
    "recovery_hold_ticks_m": "M",
    "useful_production_ticks_t": "T",
    "unanswered_request_ticks_d": "D",
}


def _parameter_value(value: Any) -> str:
    """A parameter in the locale: every decimal it was set with, and no more."""
    if isinstance(value, bool) or value is None:
        return str(value)
    try:
        number = Decimal(str(value))
    except ArithmeticError, ValueError:
        return str(value)
    if not number.is_finite():
        return str(value)
    exponent = number.normalize().as_tuple().exponent
    places = max(0, -exponent) if isinstance(exponent, int) else 0
    return format_number(number, places)
