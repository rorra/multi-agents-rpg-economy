"""Bilingual report generation (design v1.0 s11; IMPLEMENTATION.md section 2).

Both languages render from one results file. Nothing about a language is
stored: the file holds enum values and numbers, and every string is applied here
through ``i18n.t``, so ``make report LOCALE=es`` and ``LOCALE=en`` describe the
same numbers.
"""

from __future__ import annotations

import json
import statistics
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from rpg_economy.analysis import figures
from rpg_economy.analysis.results import COMMON_HORIZON_METRICS, Results, applicable_metrics
from rpg_economy.i18n import set_locale, t
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


def render_report(results: Results, *, locale: str, figure_paths: dict[str, Path]) -> str:
    """Render the whole report in one language."""
    set_locale(locale)
    target_label = good_label(
        next((run.target_good for run in results.runs if run.target_good), None)
    )

    mock = any(run.mock for run in results.runs)
    lines: list[str] = [
        t("report.title"),
        t(
            "report.subtitle",
            scenario=t(f"scenario.{results.scenario}"),
            experiment=t(f"experiment.{results.experiment}"),
            seeds=len(results.seeds),
        ),
    ]
    if mock:
        # Mock runs exercise the pipeline; they say nothing about a model.
        lines += ["", t("report.note.mock_runs")]
    if results.pilot_seeds_included:
        # Pilot seeds are never used for inference (design s11.4): the numbers
        # stand, but as exploration, not as a test of anything.
        lines += ["", t("report.note.pilot_seeds")]
    if results.metric_parameters:
        # Not part of any run's identity, so the report says which were applied.
        lines += ["", t("report.note.metric_parameters", params=_metric_parameters(results))]

    # -- per-condition medians, primary metrics first (design s11.3) ------
    all_metrics = list(applicable_metrics(results.runs))
    primary = [m for m in results.primary_metrics if m in all_metrics]
    # The common-horizon metrics get their own section below (design s8.4).
    common = [m for m in all_metrics if m in COMMON_HORIZON_METRICS]
    secondary = [m for m in all_metrics if m not in primary and m not in common]
    lines += ["", t("report.section.primary", target=target_label)]
    lines += _median_table(results, primary)
    lines += ["", t("report.section.secondary")]
    lines += _median_table(results, secondary)
    lines += ["", t("report.note.medians_pairable")]
    lines += _incomplete_trios_note(results)
    lines += _skipped_note(results)
    lines += _rejections_note(results)

    # -- measurability, where anything was missing ------------------------
    if results.measurability:
        lines += ["", t("report.section.measurability"), t("report.note.measurability"), ""]
        lines += _table(
            [
                t("report.col.metric"),
                *(t(f"condition.{c}") for c in results.conditions),
            ],
            [
                [
                    metric_label(metric),
                    *(
                        t(
                            "report.value.measured",
                            measured=census.get(c, {}).get("measured", 0),
                            total=census.get(c, {}).get("runs", 0),
                        )
                        for c in results.conditions
                    ),
                ]
                for metric, census in sorted(results.measurability.items())
            ],
        )

    # -- measurability as a companion outcome (design s11.3) ----------
    outcomes = results.companion_outcomes
    if outcomes:
        lines += ["", t("report.section.companion_outcome"), t("report.note.companion_outcome"), ""]
        lines += _table(
            [t("report.col.outcome"), *(t(f"condition.{c}") for c in results.conditions)],
            [
                [
                    t("report.metric.measurable", metric=metric_label(metric)),
                    *(
                        t(
                            "report.value.measured",
                            measured=outcome["counts"].get(c, {}).get("measurable", 0),
                            total=outcome["counts"].get(c, {}).get("runs", 0),
                        )
                        for c in results.conditions
                    ),
                ]
                for metric, outcome in outcomes.items()
            ],
        )
        rows = [
            [
                metric_label(metric),
                contrast_label(row),
                _fmt(row["discordant_pairs"], 0),
                _p(row["p_value"]),
                _p(row.get("p_value_holm")),
            ]
            for metric, outcome in outcomes.items()
            for row in outcome["paired_binary"]
        ]
        if rows:
            lines += ["", t("report.test.paired_binary_measurable")]
            lines += _table(
                [
                    t("report.col.metric"),
                    t("report.col.contrast"),
                    t("report.col.discordant"),
                    t("report.col.p"),
                    t("report.col.p_holm"),
                ],
                rows,
            )

    # -- contrasts --------------------------------------------------------
    lines += ["", t("report.note.primary_contrast"), t("report.note.holm")]

    ordered = sorted(
        set(results.contrasts) | set(results.suppressed_contrasts),
        key=lambda metric: (
            metric not in primary,
            all_metrics.index(metric) if metric in all_metrics else len(all_metrics),
        ),
    )
    for metric in ordered:
        contrasts = results.contrasts.get(metric, [])
        suppressed = results.suppressed_contrasts.get(metric)
        if not contrasts and not suppressed:
            continue
        lines += ["", t("report.section.contrasts_for", metric=metric_label(metric))]
        if suppressed:
            lines += _suppression_note(suppressed)
        if not contrasts:
            continue
        lines += _table(
            [
                t("report.col.contrast"),
                t("report.col.n"),
                t("report.col.dropped"),
                t("report.col.median_diff"),
                t("report.col.effect"),
                t("report.col.p"),
                t("report.col.p_holm"),
                t("report.col.ci"),
            ],
            [
                [
                    contrast_label(c),
                    # Pairs used and pairs dropped, per contrast (design s11.3):
                    # each pair of conditions loses its own seeds.
                    _fmt(c["n_pairs"], 0),
                    _fmt(c.get("n_dropped", 0), 0),
                    _fmt(c["median_difference"]),
                    _fmt(c["rank_biserial"], 2),
                    _p(c["p_value"]),
                    _p(c["p_value_holm"]),
                    (
                        f"[{_fmt(c['ci_low'])}; {_fmt(c['ci_high'])}]"
                        if c["ci_low"] is not None
                        else t("report.value.missing")
                    ),
                ]
                for c in contrasts
            ],
        )
        if any(c["n_pairs"] == 0 for c in contrasts):
            lines.append(t("report.note.no_pairs"))
        if any("few_pairs" in c["note"] for c in contrasts):
            lines.append(t("report.note.few_pairs"))
        if any(c.get("n_dropped") for c in contrasts):
            lines.append(t("report.note.dropped_pairs_column"))

    # -- recovery, censored ----------------------------------------------
    survival = results.survival
    lines += ["", t("report.section.recovery"), t("report.note.censored_excluded"), ""]
    lines += _recovery_table(results, survival)

    lines += _recovery_tests(survival)

    lines += _common_horizon_section(results, common)

    lines += _propagation_section(results)
    lines += _volume_by_good_section(results)

    lines += _system_section(results)

    if figure_paths:
        lines += ["", *(f"[{name}] {path.name}" for name, path in sorted(figure_paths.items()))]

    return "\n".join(lines)


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


def _skipped_note(results: Results) -> list[str]:
    """What the whole analysis left out, not only this cell (design s11.3).

    Seeds outside the analysed seed set, and other cells with no complete trio,
    each with the conditions its logs did cover.
    """
    lines: list[str] = []
    if results.skipped_seeds:
        seeds = ", ".join(str(seed) for seed in sorted(results.skipped_seeds))
        lines.append(
            t("report.note.skipped_seeds", seeds=seeds, count=_fmt(len(results.skipped_seeds), 0))
        )
    if results.skipped_cells:
        cells = "; ".join(
            f"{_cell_label(cell)} ({', '.join(conditions)})"
            for cell, conditions in sorted(results.skipped_cells.items())
        )
        lines.append(
            t("report.note.skipped_cells", cells=cells, count=_fmt(len(results.skipped_cells), 0))
        )
    return lines


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


def _metric_parameters(results: Results) -> str:
    """``W = 5, X = 0.1, ...``: the design's letters first, anything else by name."""
    parameters = results.metric_parameters
    ordered = [name for name in METRIC_PARAMETER_LETTERS if name in parameters]
    ordered += sorted(name for name in parameters if name not in METRIC_PARAMETER_LETTERS)
    return ", ".join(
        f"{METRIC_PARAMETER_LETTERS.get(name, name)} = {_parameter_value(parameters[name])}"
        for name in ordered
    )


def _rejections_note(results: Results) -> list[str]:
    """Logs, trios and cells the checks left out of the whole analysis (design s12)."""
    lines: list[str] = []
    if results.skipped_logs:
        reasons: dict[str, int] = {}
        for entry in results.skipped_logs:
            reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
        summary = "; ".join(
            f"{t(f'report.skip_reason.{reason}')}: {_fmt(count, 0)}"
            for reason, count in sorted(reasons.items())
        )
        lines.append(
            t(
                "report.note.skipped_logs",
                reasons=summary,
                count=_fmt(len(results.skipped_logs), 0),
            )
        )
    rejected = [
        (cell, entry)
        for cell, entries in sorted(results.rejected_trios.items())
        for entry in entries
    ]
    if rejected:
        trios = "; ".join(
            f"{_cell_label(cell)} {entry['seed']} ({t('report.reject_reason.' + entry['reason'])})"
            for cell, entry in rejected
        )
        lines.append(t("report.note.rejected_trios", trios=trios, count=_fmt(len(rejected), 0)))
    if results.refused_cells:
        cells = "; ".join(
            f"{_cell_label(cell)} ({', '.join(fields)})"
            for cell, fields in sorted(results.refused_cells.items())
        )
        lines.append(
            t("report.note.refused_cells", cells=cells, count=_fmt(len(results.refused_cells), 0))
        )
    return lines


def _collation_key(label: str) -> str:
    """Case- and accent-insensitive: "Árbol" sorts with "arco", not after "zinc"."""
    decomposed = unicodedata.normalize("NFKD", label.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _cell_label(cell: str) -> str:
    """A ``scenario/experiment`` key in the reader's language."""
    scenario, _, experiment = cell.partition("/")
    return f"{t(f'scenario.{scenario}')} / {t(f'experiment.{experiment}')}"


def _common_horizon_section(results: Results, metrics: list[str]) -> list[str]:
    """Every economic metric at the trio's common active horizon (design s8.4, s11.2).

    Medians over the pairable seeds, then the recoveries censored at that
    horizon: the target's and, where there is one, the propagated product's.
    The contrasts of these metrics are with the others, under their own names.
    """
    recovery = results.survival_common_horizon
    propagated = results.survival_propagation_common_horizon
    if not metrics and recovery is None and propagated is None:
        return []
    lines = ["", t("report.section.common_horizon"), t("report.note.common_horizon"), ""]
    lines += _median_table(results, metrics)
    if recovery is not None:
        lines += ["", t("report.section.recovery_common"), ""]
        lines += _recovery_table(results, recovery)
        lines += _recovery_tests(recovery)
    block = next((run.propagation for run in results.runs if run.propagation), None)
    if propagated is not None and block is not None:
        lines += [
            "",
            t(
                "report.section.propagation_common",
                good=good_label(block["good"]),
                resource=good_label(block["source"]),
            ),
            "",
        ]
        lines += _recovery_table(results, propagated)
        lines += _recovery_tests(propagated)
    return lines


def _propagation_section(results: Results) -> list[str]:
    """The shock's propagation to the main finished product (secondary, s11.3)."""
    if results.survival_propagation is None:
        return []
    block = next((run.propagation for run in results.runs if run.propagation), None)
    if block is None:
        return []
    lines = [
        "",
        t(
            "report.section.propagation",
            good=good_label(block["good"]),
            resource=good_label(block["source"]),
        ),
        t("report.note.propagation"),
        "",
    ]
    lines += _recovery_table(results, results.survival_propagation)
    return lines + _recovery_tests(results.survival_propagation)


def _volume_by_good_section(results: Results) -> list[str]:
    """Median units traded per good, shock phase against base phase (s11.1)."""
    summary = results.volume_by_good
    conditions = [c for c in results.conditions if c in summary]
    goods = sorted({good for c in conditions for good in summary[c]})
    if not goods:
        return []

    def cell(condition: str, good: str) -> str:
        phases = summary[condition].get(good, {})
        shock = (phases.get("shock") or {}).get("units")
        base = (phases.get("base") or {}).get("units")
        return f"{_fmt(shock, 1)} / {_fmt(base, 1)}"

    rows = [
        [good_label(good), *(cell(c, good) for c in conditions)]
        for good in goods
        if any(
            (summary[c].get(good, {}).get(phase) or {}).get("units")
            for c in conditions
            for phase in ("base", "shock")
        )
    ]
    # In the reader's alphabet: sorted by the localized name, not the enum key.
    rows.sort(key=lambda row: (_collation_key(row[0]), row[0]))
    if not rows:
        return []
    return [
        "",
        t("report.section.volume_by_good"),
        "",
        *_table([t("report.col.good"), *(t(f"condition.{c}") for c in conditions)], rows),
    ]


def _system_section(results: Results) -> list[str]:
    """Descriptive coordination and cost figures per condition (design s11.2)."""
    if not results.system:
        return []
    conditions = [c for c in results.conditions if c in results.system]
    kinds = sorted({kind for c in conditions for kind in results.system[c]["messages_by_type"]})

    def row(label: str, render) -> list[str]:
        return [label, *(render(results.system[c]) for c in conditions)]

    buckets = sorted(
        {b for c in conditions for b in results.system[c].get("tokens_by_bucket_median", {})}
    )

    def per_bucket(field: str, key: str) -> list[list[str]]:
        """One row per bucket that any condition used, for one per-actor figure."""
        used = sorted(
            {
                bucket
                for c in conditions
                for bucket, value in results.system[c].get(field, {}).items()
                if value
            }
        )
        return [
            row(
                t(key, bucket=t(f"bucket.{bucket}")),
                lambda s, bucket=bucket: _fmt(s.get(field, {}).get(bucket), 0),
            )
            for bucket in used
        ]

    tasks = sorted({k for c in conditions for k in results.system[c].get("task_types", {})})
    rows = [
        row(t("report.system.activations"), lambda s: _fmt(s.get("activations_median"), 0)),
        row(t("report.system.wasted"), lambda s: _fmt(s.get("wasted_median"), 0)),
        row(t("report.system.k_binding"), lambda s: _percent(s.get("k_binding_rate_median"))),
        row(t("report.system.invalid"), lambda s: _fmt(s.get("invalid_outputs", 0), 0)),
        row(t("report.system.truncated"), lambda s: _fmt(s.get("truncated_outputs", 0), 0)),
        *(
            row(
                t("report.system.tokens_bucket", bucket=t(f"bucket.{bucket}")),
                lambda s, bucket=bucket: _fmt(s.get("tokens_by_bucket_median", {}).get(bucket), 0),
            )
            for bucket in buckets
        ),
        # Per-actor detail (design s11.2): input and output are priced
        # differently, and calls show where the tokens were spent.
        *per_bucket("input_tokens_by_bucket_median", "report.system.input_tokens_bucket"),
        *per_bucket("output_tokens_by_bucket_median", "report.system.output_tokens_bucket"),
        *per_bucket("calls_by_bucket_median", "report.system.calls_bucket"),
        row(
            t("report.system.budget_fraction"), lambda s: _percent(s.get("budget_fraction_median"))
        ),
        *(
            row(
                t("report.system.task_type", task=t(f"task.{task}")),
                lambda s, task=task: _fmt(s.get("task_types", {}).get(task, 0), 0),
            )
            for task in tasks
        ),
        *(
            row(
                t("report.system.messages", kind=t(f"message.{kind}")),
                lambda s, kind=kind: _fmt(s["messages_by_type"].get(kind, 0), 0),
            )
            for kind in kinds
        ),
        row(t("report.system.unanswered"), lambda s: _fmt(s["unanswered_requests"], 0)),
        row(t("report.system.latency"), lambda s: _fmt(s["latency_mean"], 2)),
        row(t("report.system.director_retries"), lambda s: _fmt(s["director_retries"], 0)),
        row(t("report.system.closed_by_cap"), lambda s: _fmt(s["cycles_closed_by_cap"], 0)),
        row(t("report.system.director_error_rate"), lambda s: _percent(s["director_error_rate"])),
        row(
            t("report.system.exhausted"),
            lambda s: t("report.value.measured", measured=s["exhausted_runs"], total=s["runs"]),
        ),
        row(t("report.system.exhaustion_tick"), lambda s: _fmt(s["exhaustion_tick_median"], 1)),
    ]
    return [
        "",
        t("report.section.coordination_metrics"),
        "",
        *_table([t("report.col.metric"), *(t(f"condition.{c}") for c in conditions)], rows),
    ]


def generate_report(
    results: Results,
    destination: Path | str,
    *,
    locale: str,
    price_series: dict[str, list[float | None]] | None = None,
    shock_tick: int = 0,
    reversion_tick: int = 0,
) -> Path:
    """Write one language's report and its figures.

    Reports go to ``reports/<locale>/`` so the two languages never overwrite
    each other (IMPLEMENTATION.md section 2).
    """
    set_locale(locale)
    out = Path(destination) / locale
    out.mkdir(parents=True, exist_ok=True)

    target_label = good_label(
        next((run.target_good for run in results.runs if run.target_good), None)
    )
    # liquidity_loss measures traded value, not a price (design s11.3): the
    # figure must say so rather than plot value under a price axis.
    traded_value = any(run.target_kind == "traded_value" for run in results.runs)

    produced: dict[str, Path] = {}
    if price_series:
        produced["price"] = figures.price_trajectory(
            price_series,
            good_label=target_label,
            shock_tick=shock_tick,
            reversion_tick=reversion_tick,
            path=out / "price_trajectory.png",
            traded_value=traded_value,
        )

    # No curves when a recovery pair is withheld, for the same reason.
    if results.survival.get("kaplan_meier") and not results.survival.get("suppressed_pairs"):
        produced["km"] = figures.kaplan_meier_curves(
            results.survival["kaplan_meier"], path=out / "recovery_km.png"
        )

    # The paired figure shows the first primary metric that was actually
    # contrasted. A suppressed contrast gets no figure: drawing its surviving
    # pairs would show exactly the comparison the suppression withholds.
    primary = next(
        (
            m
            for m in results.primary_metrics
            if any(c["left"] == "A" and c["right"] == "B" for c in results.contrasts.get(m, []))
        ),
        None,
    )
    table = results.by_condition(primary) if primary else {}
    if primary and "A" in table and "B" in table:
        shared = sorted(set(table["A"]) & set(table["B"]))
        if shared:
            produced["paired"] = figures.paired_differences(
                {seed: table["A"][seed] - table["B"][seed] for seed in shared},
                metric_label=metric_label(primary),
                path=out / "paired_differences.png",
            )

    # A figure an earlier render left and this one withheld must not linger.
    for stale in ("recovery_km.png", "paired_differences.png", "price_trajectory.png"):
        path = out / stale
        if path.exists() and path not in produced.values():
            path.unlink()

    text = render_report(results, locale=locale, figure_paths=produced)
    report_path = out / "report.txt"
    report_path.write_text(text + "\n", encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(results.to_payload(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report_path


def generate_all_locales(
    results: Results, destination: Path | str, **kwargs: Any
) -> dict[str, Path]:
    """Render every supported language from the same results."""
    from rpg_economy.i18n import SUPPORTED_LOCALES, get_catalog

    active = get_catalog().locale
    try:
        return {
            locale: generate_report(results, destination, locale=locale, **kwargs)
            for locale in SUPPORTED_LOCALES
        }
    finally:
        set_locale(active)  # rendering must not change the caller's language
