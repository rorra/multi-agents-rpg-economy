"""Figures for the paper, drawn from finished run logs and the report's results.

The report (``analysis.report``) carries the pre-registered contrasts and their
three figures. This module adds the descriptive pictures a write-up needs: how
prices, volumes, flows and wealth moved through base, shock and recovery in
each condition, what each condition spent in tokens and calls, how the agents
acted, and the per-seed spread behind each paired contrast.

Two inputs, both read only: the run logs (time series the results file does not
keep) and ``results.json`` written by ``rpg-economy report`` (the per-run
metrics and the Kaplan-Meier estimates, so nothing here recomputes a metric the
report owns). Every label resolves through ``i18n.t``, and each figure is
written once per supported language, as PNG and PDF.

Across seeds a series is summarised by its median with the interquartile range
as a band: ten seeds are too few for a mean to say more than the median, and
the band shows the spread a single line would hide. A window with fewer than
``MIN_SEEDS`` runs carrying a value is left empty rather than drawn from one or
two thin markets.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpg_economy.analysis.figures import (
    CONDITION_COLOURS,
    CONDITION_LINESTYLES,
    GRID,
    INK,
    INK_MUTED,
    _figure,
    _style,
)
from rpg_economy.config.models import SimulationConfig
from rpg_economy.i18n import format_number, t

CONDITIONS = ("A", "B", "C")
RESOURCES = ("iron", "wood", "leather", "herbs", "arcane_crystal")
PRODUCTS = ("melee_weapon", "armor", "bow_gear", "staff_focus", "health_potion", "mana_potion")
GOODS = RESOURCES + PRODUCTS
PHASES = ("base", "shock", "recovery")
TRIGGERS = (
    "T1_message_received",
    "T2_order_event",
    "T3_price_move",
    "T4_own_state_threshold",
    "T5_inactivity",
)
TASKS = (
    "ANALYZE_MARKET",
    "EVALUATE_EXPEDITION",
    "EVALUATE_PRODUCTION",
    "MANAGE_PROCUREMENT",
    "MANAGE_INVENTORY",
    "ADDRESS_IMBALANCE",
)
ACTIONS = (
    "buy",
    "sell",
    "place_orders",
    "cancel_order",
    "produce",
    "start_expedition",
    "equip",
    "send_message",
    "wait",
)
#: Ticks per point of a time series.
WINDOW = 10
#: Fewest runs with a value for a window's median to be drawn.
MIN_SEEDS = 3
FORMATS = ("png", "pdf")


# ---------------------------------------------------------------------------
# Extraction: one pass over each log
# ---------------------------------------------------------------------------


@dataclass
class RunSeries:
    """What one run log contributes to the figures, language-neutral."""

    condition: str
    seed: int
    windows: int
    #: good -> per window mean trade price (None when nothing traded).
    price: dict[str, list[float | None]] = field(default_factory=dict)
    #: per window units traded, all goods.
    units: list[int] = field(default_factory=list)
    #: per window traded value, all goods.
    value: list[float] = field(default_factory=list)
    #: role -> per window mean gold of that role's agents (last observed).
    gold: dict[str, list[float | None]] = field(default_factory=dict)
    #: per window Gini of every agent's last observed gold.
    gini: list[float | None] = field(default_factory=list)
    #: phase -> flow -> good -> units.
    flows: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    #: phase -> started / succeeded expeditions.
    expeditions: dict[str, dict[str, int]] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)
    triggers: dict[str, int] = field(default_factory=dict)
    tasks: dict[str, int] = field(default_factory=dict)
    activations: int = 0
    wasted: int = 0
    invalid: int = 0
    #: kind ("input"/"output") -> bucket -> tokens.
    tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    #: role -> final gold of each agent.
    final_gold: dict[str, list[float]] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return self.__dict__.copy()


def gini(values: list[float]) -> float | None:
    """Gini coefficient of non-negative holdings (0 equal, towards 1 concentrated)."""
    xs = sorted(max(0.0, v) for v in values)
    n, total = len(xs), sum(xs)
    if n < 2 or total <= 0:
        return None
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * weighted) / (n * total) - (n + 1) / n


def extract(path: Path, shock_tick: int, reversion_tick: int, horizon: int) -> RunSeries:
    """Read one log once and keep only what the figures draw."""
    windows = (horizon + WINDOW - 1) // WINDOW

    def phase(tick: int) -> str:
        if tick < shock_tick:
            return "base"
        return "shock" if tick < reversion_tick else "recovery"

    prices: dict[str, list[list[float]]] = {g: [[] for _ in range(windows)] for g in GOODS}
    units = [0] * windows
    value = [0.0] * windows
    flows = {p: defaultdict(Counter) for p in PHASES}
    expeditions = {p: Counter() for p in PHASES}
    actions: Counter = Counter()
    triggers: Counter = Counter()
    tasks: Counter = Counter()
    roles: dict[str, str] = {}
    last_gold: dict[str, float] = {}
    gold_by_window: dict[str, list[float | None]] = {
        "adventurer": [None] * windows,
        "artisan": [None] * windows,
    }
    gini_by_window: list[float | None] = [None] * windows
    activations = wasted = invalid = 0
    condition, seed = "", 0
    tokens: dict[str, dict[str, int]] = {}
    calls: dict[str, int] = {}
    final_gold: dict[str, list[float]] = {}
    current_window = 0

    def close_window(w: int) -> None:
        for role in gold_by_window:
            held = [g for a, g in last_gold.items() if roles.get(a) == role]
            gold_by_window[role][w] = statistics.fmean(held) if held else None
        gini_by_window[w] = gini(list(last_gold.values()))

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            kind = event["type"]
            tick = event.get("tick") or 0
            payload = event.get("payload") or {}
            w = min(tick // WINDOW, windows - 1)
            while current_window < w:
                close_window(current_window)
                current_window += 1
            match kind:
                case "run_metadata":
                    condition, seed = payload["condition"], int(payload["seed"])
                case "trade_executed":
                    prices[payload["good"]][w].append(float(payload["price"]))
                    units[w] += int(payload["quantity"])
                    value[w] += float(payload["value"])
                    if payload.get("buyer_role") == "adventurer" and payload["good"] in PRODUCTS:
                        flows[phase(tick)]["bought"][payload["good"]] += int(payload["quantity"])
                case "expedition_started":
                    expeditions[phase(tick)]["started"] += 1
                case "expedition_resolved":
                    if payload.get("outcome") == "success":
                        expeditions[phase(tick)]["succeeded"] += 1
                    for good, qty in (payload.get("yielded") or {}).items():
                        flows[phase(tick)]["gathered"][good] += int(qty)
                case "production_completed":
                    flows[phase(tick)]["made"][payload["product"]] += int(payload["quantity"])
                    for good, qty in payload["consumed"].items():
                        flows[phase(tick)]["used"][good] += int(qty)
                case "consumption":
                    flows[phase(tick)]["consumed"][payload["product"]] += int(payload["quantity"])
                case "observation_delivered":
                    obs = payload["observation"]
                    roles[obs["agent_id"]] = obs["role"]
                    last_gold[obs["agent_id"]] = float(obs["gold"])
                case "agent_output":
                    activations += 1
                case "activation_wasted":
                    wasted += 1
                case "agent_output_invalid":
                    invalid += 1
                case "action_executed":
                    actions[payload["action"]] += 1
                case "trigger_fired":
                    triggers[payload["trigger"]] += 1
                case "director_task_assigned":
                    tasks[payload["type"]] += 1
                case "tokens_consumed":
                    tokens = {"input": dict(payload["input"]), "output": dict(payload["output"])}
                    calls = dict(payload["calls"])
                case "run_end":
                    for agent, held in payload["holdings"].items():
                        final_gold.setdefault(held["role"], []).append(float(held["gold"]))
                        roles[agent] = held["role"]
                        last_gold[agent] = float(held["gold"])
    while current_window < windows:
        close_window(current_window)
        current_window += 1

    return RunSeries(
        condition=condition,
        seed=seed,
        windows=windows,
        price={g: [statistics.fmean(v) if v else None for v in prices[g]] for g in GOODS},
        units=units,
        value=value,
        gold=gold_by_window,
        gini=gini_by_window,
        flows={p: {k: dict(v) for k, v in flows[p].items()} for p in PHASES},
        expeditions={p: dict(expeditions[p]) for p in PHASES},
        actions=dict(actions),
        triggers=dict(triggers),
        tasks=dict(tasks),
        activations=activations,
        wasted=wasted,
        invalid=invalid,
        tokens=tokens,
        calls=calls,
        final_gold=final_gold,
    )


def extract_all(logs: list[Path], config: SimulationConfig, *, workers: int = 8) -> list[RunSeries]:
    """Extract every log, in parallel: each is an independent read."""
    args = (config.phases.shock_tick, config.phases.reversion_tick, config.time.horizon_ticks)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(extract, path, *args) for path in logs]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Summaries across seeds
# ---------------------------------------------------------------------------


def _quantiles(values: list[float]) -> tuple[float, float, float] | None:
    if len(values) < MIN_SEEDS:
        return None
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, median, q3


def band(
    series: list[list[float | None]],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Per window median and interquartile range across runs."""
    width = max((len(s) for s in series), default=0)
    lo: list[float | None] = []
    mid: list[float | None] = []
    hi: list[float | None] = []
    for w in range(width):
        q = _quantiles([s[w] for s in series if w < len(s) and s[w] is not None])
        lo.append(q[0] if q else None)
        mid.append(q[1] if q else None)
        hi.append(q[2] if q else None)
    return lo, mid, hi


def by_condition(runs: list[RunSeries]) -> dict[str, list[RunSeries]]:
    grouped: dict[str, list[RunSeries]] = {c: [] for c in CONDITIONS}
    for run in runs:
        grouped.setdefault(run.condition, []).append(run)
    for group in grouped.values():
        group.sort(key=lambda r: r.seed)
    return grouped


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _save(fig, out: Path, name: str, *, layout: bool = True) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    if layout:  # figures that reserve room for a figure-level legend lay out themselves
        fig.tight_layout()
    paths = []
    for fmt in FORMATS:
        path = out / f"{name}.{fmt}"
        fig.savefig(path, facecolor="white", dpi=200)
        paths.append(path)
    _figure().close(fig)
    return paths


def _shade(ax, config: SimulationConfig) -> None:
    """The shock interval as a recessive band, in tick units."""
    ax.axvspan(
        config.phases.shock_tick, config.phases.reversion_tick, color=GRID, alpha=0.7, linewidth=0
    )


def _legend(ax, *, below: bool = True, ncols: int = 3) -> None:
    if not ax.get_legend_handles_labels()[0]:
        return
    if below:
        ax.legend(
            frameon=False,
            fontsize=8,
            labelcolor=INK_MUTED,
            ncols=ncols,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
        )
    else:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED)


def _title(ax, key: str, **kwargs: Any) -> None:
    ax.set_title(t(key, **kwargs), fontsize=10.5, loc="left", color=INK)


def _ticks(width: int) -> list[float]:
    """Window index to the tick at its centre."""
    return [w * WINDOW + WINDOW / 2 for w in range(width)]


def _band_lines(ax, grouped: dict[str, list[list[float | None]]], *, label: bool = True) -> None:
    for condition in CONDITIONS:
        series = grouped.get(condition) or []
        if not series:
            continue
        lo, mid, hi = band(series)
        xs = _ticks(len(mid))
        colour = CONDITION_COLOURS[condition]
        pts = [(x, a, b, c) for x, a, b, c in zip(xs, lo, mid, hi, strict=True) if b is not None]
        if not pts:
            continue
        ax.fill_between(
            [p[0] for p in pts],
            [p[1] for p in pts],
            [p[3] for p in pts],
            color=colour,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            [p[0] for p in pts],
            [p[2] for p in pts],
            color=colour,
            linestyle=CONDITION_LINESTYLES[condition],
            linewidth=1.9,
            label=t(f"condition.{condition}") if label else None,
        )


def good_label(good: str) -> str:
    return t(f"resource.{good}") if good in RESOURCES else t(f"product.{good}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_price(
    runs: list[RunSeries], config: SimulationConfig, good: str, out: Path, name: str
) -> list[Path]:
    """One good's trade price over time: median and IQR across seeds, guide price dashed."""
    plt = _figure()
    fig, ax = plt.subplots(figsize=(8, 4.0))
    _style(ax)
    _shade(ax, config)
    grouped = by_condition(runs)
    _band_lines(ax, {c: [r.price[good] for r in rs] for c, rs in grouped.items()})
    guide = config.economy.initial_reference_price.get(good)
    if guide is not None:
        ax.axhline(
            float(guide),
            color=INK_MUTED,
            linewidth=1,
            linestyle=(0, (4, 3)),
            label=t("paper.label.guide_price"),
        )
    _title(ax, "paper.figure.price", good=good_label(good))
    ax.set_xlabel(t("axis.tick"))
    ax.set_ylabel(t("axis.price"))
    _legend(ax, ncols=4)
    return _save(fig, out, name)


def fig_price_index(runs: list[RunSeries], config: SimulationConfig, out: Path) -> list[Path]:
    """Every good's price relative to its guide price, averaged over the goods traded."""
    plt = _figure()
    fig, ax = plt.subplots(figsize=(8, 4.0))
    _style(ax)
    _shade(ax, config)
    guide = {g: float(config.economy.initial_reference_price[g]) for g in GOODS}

    def index(run: RunSeries) -> list[float | None]:
        out_: list[float | None] = []
        for w in range(run.windows):
            ratios = [run.price[g][w] / guide[g] for g in GOODS if run.price[g][w] is not None]
            out_.append(statistics.fmean(ratios) if ratios else None)
        return out_

    grouped = by_condition(runs)
    _band_lines(ax, {c: [index(r) for r in rs] for c, rs in grouped.items()})
    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    _title(ax, "paper.figure.price_index")
    ax.set_xlabel(t("axis.tick"))
    ax.set_ylabel(t("paper.axis.price_over_guide"))
    _legend(ax)
    return _save(fig, out, "03_price_index_all_goods")


def fig_price_grid(runs: list[RunSeries], config: SimulationConfig, out: Path) -> list[Path]:
    """Small multiples: every good's median price, one panel per good."""
    plt = _figure()
    fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharex=True)
    grouped = by_condition(runs)
    for ax, good in zip(axes.flat, GOODS, strict=False):
        _style(ax)
        _shade(ax, config)
        _band_lines(
            ax,
            {c: [r.price[good] for r in rs] for c, rs in grouped.items()},
            label=good == GOODS[0],
        )
        ax.axhline(
            float(config.economy.initial_reference_price[good]),
            color=INK_MUTED,
            linewidth=0.8,
            linestyle=(0, (4, 3)),
        )
        ax.set_title(good_label(good), fontsize=9, loc="left", color=INK)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(
        handles, labels, frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="center"
    )
    fig.suptitle(t("paper.figure.price_grid"), x=0.02, ha="left", fontsize=11, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel(t("axis.tick"))
    return _save(fig, out, "04_price_every_good")


def fig_seed_panels(
    runs: list[RunSeries], config: SimulationConfig, good: str, out: Path, name: str
) -> list[Path]:
    """The shocked good's price in every seed, the three conditions overlaid."""
    plt = _figure()
    grouped = by_condition(runs)
    seeds = sorted({r.seed for r in runs})
    cols = 5
    rows = (len(seeds) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), sharex=True, sharey=True)
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, seed in zip(flat, seeds, strict=False):
        _style(ax)
        _shade(ax, config)
        for condition in CONDITIONS:
            run = next((r for r in grouped.get(condition, []) if r.seed == seed), None)
            if run is None:
                continue
            series = run.price[good]
            pts = [
                (x, v) for x, v in zip(_ticks(len(series)), series, strict=True) if v is not None
            ]
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=CONDITION_COLOURS[condition],
                linestyle=CONDITION_LINESTYLES[condition],
                linewidth=1.4,
                label=t(f"condition.{condition}") if seed == seeds[0] else None,
            )
        ax.set_title(t("paper.label.seed", seed=seed), fontsize=9, loc="left", color=INK)
    for ax in flat[len(seeds) :]:
        ax.axis("off")
    fig.suptitle(
        t("paper.figure.seed_panels", good=good_label(good)),
        x=0.02,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.legend(
        *flat[0].get_legend_handles_labels(),
        frameon=False,
        fontsize=9,
        labelcolor=INK_MUTED,
        ncols=3,
        loc="lower center",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return _save(fig, out, name, layout=False)


def fig_volume(runs: list[RunSeries], config: SimulationConfig, out: Path) -> list[Path]:
    """Units traded per window, all goods."""
    plt = _figure()
    fig, ax = plt.subplots(figsize=(8, 4.0))
    _style(ax)
    _shade(ax, config)
    grouped = by_condition(runs)
    _band_lines(ax, {c: [[float(u) for u in r.units] for r in rs] for c, rs in grouped.items()})
    _title(ax, "paper.figure.volume", window=WINDOW)
    ax.set_xlabel(t("axis.tick"))
    ax.set_ylabel(t("paper.axis.units"))
    _legend(ax)
    return _save(fig, out, "07_traded_units")


def _grouped_bars(
    ax, categories: list[str], values: dict[str, list[float]], labels: list[str]
) -> None:
    width = 0.8 / len(CONDITIONS)
    for i, condition in enumerate(CONDITIONS):
        xs = [k + (i - 1) * width for k in range(len(categories))]
        ax.bar(
            xs,
            values[condition],
            width=width * 0.95,
            color=CONDITION_COLOURS[condition],
            label=t(f"condition.{condition}"),
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(labels, fontsize=8.5)


def fig_flows(runs: list[RunSeries], out: Path, good: str, product: str) -> list[Path]:
    """Herbs gathered and used, and potions made and bought, per phase (median per run)."""
    plt = _figure()
    fig, grid = plt.subplots(2, 2, figsize=(10, 7))
    axes = list(grid.flat)
    grouped = by_condition(runs)
    panels = (
        (good, "gathered", "paper.flow.gathered"),
        (good, "used", "paper.flow.used"),
        (product, "made", "paper.flow.made"),
        (product, "bought", "paper.flow.bought"),
    )
    labels = [t(f"phase.{p}") for p in PHASES]
    for ax, (item, kind, key) in zip(axes, panels, strict=True):
        _style(ax)
        values = {
            c: [
                _median([r.flows[p].get(kind, {}).get(item, 0) for r in grouped[c]]) for p in PHASES
            ]
            for c in CONDITIONS
        }
        _grouped_bars(ax, list(PHASES), values, labels)
        ax.set_title(
            t("paper.label.flow_panel", item=good_label(item), flow=t(key)),
            fontsize=9.5,
            loc="left",
            color=INK,
        )
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(t("paper.axis.units_per_run"))
    fig.suptitle(t("paper.figure.flows"), x=0.02, ha="left", fontsize=11, color=INK)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        fontsize=9,
        labelcolor=INK_MUTED,
        ncols=3,
        loc="lower center",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return _save(fig, out, "08_flows_by_phase", layout=False)


def fig_production(runs: list[RunSeries], out: Path) -> list[Path]:
    """Every product made per phase, median per run, one panel per product."""
    plt = _figure()
    fig, grid = plt.subplots(2, 3, figsize=(12, 6.6))
    grouped = by_condition(runs)
    labels = [t(f"phase.{p}") for p in PHASES]
    axes = list(grid.flat)
    for ax, good in zip(axes, PRODUCTS, strict=True):
        _style(ax)
        values = {
            c: [
                _median([r.flows[p].get("made", {}).get(good, 0) for r in grouped[c]])
                for p in PHASES
            ]
            for c in CONDITIONS
        }
        _grouped_bars(ax, list(PHASES), values, labels)
        ax.set_title(good_label(good), fontsize=9.5, loc="left", color=INK)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(t("paper.axis.units_per_run"))
    fig.suptitle(t("paper.figure.production"), x=0.02, ha="left", fontsize=11, color=INK)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        fontsize=9,
        labelcolor=INK_MUTED,
        ncols=3,
        loc="lower center",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return _save(fig, out, "09_production_by_phase", layout=False)


def fig_expeditions(runs: list[RunSeries], out: Path) -> list[Path]:
    """Expeditions started and their success rate per phase."""
    plt = _figure()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    grouped = by_condition(runs)
    labels = [t(f"phase.{p}") for p in PHASES]
    started = {
        c: [_median([r.expeditions[p].get("started", 0) for r in grouped[c]]) for p in PHASES]
        for c in CONDITIONS
    }
    rate = {
        c: [
            _median(
                [
                    r.expeditions[p].get("succeeded", 0) / r.expeditions[p]["started"]
                    for r in grouped[c]
                    if r.expeditions[p].get("started")
                ]
            )
            for p in PHASES
        ]
        for c in CONDITIONS
    }
    for ax, values, key, ylabel in (
        (axes[0], started, "paper.figure.expeditions_started", "paper.axis.per_run"),
        (axes[1], rate, "paper.figure.expedition_success", "paper.axis.share"),
    ):
        _style(ax)
        _grouped_bars(ax, list(PHASES), values, labels)
        _title(ax, key)
        ax.set_ylabel(t(ylabel))
    _legend(axes[0])
    return _save(fig, out, "10_expeditions")


def fig_gold(runs: list[RunSeries], config: SimulationConfig, out: Path) -> list[Path]:
    """Mean gold of adventurers and of artisans over time."""
    plt = _figure()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    grouped = by_condition(runs)
    for ax, role in zip(axes, ("adventurer", "artisan"), strict=True):
        _style(ax)
        _shade(ax, config)
        _band_lines(ax, {c: [r.gold[role] for r in rs] for c, rs in grouped.items()})
        ax.set_title(t(f"role.{role}"), fontsize=10, loc="left", color=INK)
        ax.set_xlabel(t("axis.tick"))
    axes[0].set_ylabel(t("paper.axis.mean_gold"))
    fig.suptitle(t("paper.figure.gold"), x=0.02, ha="left", fontsize=11, color=INK)
    _legend(axes[0])
    return _save(fig, out, "11_gold_by_role")


def fig_gini(runs: list[RunSeries], config: SimulationConfig, out: Path) -> list[Path]:
    plt = _figure()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _style(ax)
    _shade(ax, config)
    grouped = by_condition(runs)
    _band_lines(ax, {c: [r.gini for r in rs] for c, rs in grouped.items()})
    _title(ax, "paper.figure.gini")
    ax.set_xlabel(t("axis.tick"))
    ax.set_ylabel(t("paper.axis.gini"))
    _legend(ax)
    return _save(fig, out, "12_gini_over_time")


def _strip(
    ax, values: dict[str, list[float]], *, seeds: dict[str, list[int]] | None = None
) -> None:
    """Every run as a point per condition, its median as a bar: ten points hide nothing."""
    for i, condition in enumerate(CONDITIONS):
        ys = values.get(condition) or []
        if not ys:
            continue
        n = len(ys)
        xs = [i + (k - (n - 1) / 2) * (0.36 / max(1, n - 1)) for k in range(n)]
        ax.scatter(
            xs,
            ys,
            s=26,
            color=CONDITION_COLOURS[condition],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        median = statistics.median(ys)
        ax.hlines(median, i - 0.28, i + 0.28, color=INK, linewidth=1.6, zorder=4)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([t(f"paper.short.{c}") for c in CONDITIONS], fontsize=8)


def fig_final_gold(runs: list[RunSeries], out: Path) -> list[Path]:
    """The poorest artisan and the poorest adventurer at the end of every run."""
    plt = _figure()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    grouped = by_condition(runs)
    for ax, role in zip(axes, ("artisan", "adventurer"), strict=True):
        _style(ax)
        _strip(
            ax, {c: [min(r.final_gold.get(role, [0.0])) for r in grouped[c]] for c in CONDITIONS}
        )
        ax.axhline(0, color=INK_MUTED, linewidth=1)
        ax.set_title(
            t("paper.figure.poorest", role=t(f"role.{role}")), fontsize=10, loc="left", color=INK
        )
    axes[0].set_ylabel(t("paper.axis.final_gold"))
    return _save(fig, out, "13_poorest_agent_final_gold")


def fig_tokens(runs: list[RunSeries], out: Path) -> list[Path]:
    """Tokens per run by bucket, input and output stacked (median per run)."""
    plt = _figure()
    fig, ax = plt.subplots(figsize=(8, 4.0))
    _style(ax)
    grouped = by_condition(runs)
    segments = (
        ("agent", "input", "#2a78d6", 1.0),
        ("agent", "output", "#2a78d6", 0.55),
        ("director", "input", "#7a4fb5", 1.0),
        ("director", "output", "#7a4fb5", 0.55),
        ("bulletin", "input", "#1baf7a", 1.0),
        ("p2p_content", "input", "#eb6834", 1.0),
    )
    bottoms = [0.0] * len(CONDITIONS)
    for bucket, kind, colour, alpha in segments:
        heights = [
            _median([r.tokens.get(kind, {}).get(bucket, 0) / 1e6 for r in grouped[c]])
            for c in CONDITIONS
        ]
        if not any(heights):
            continue
        ax.bar(
            range(len(CONDITIONS)),
            heights,
            bottom=bottoms,
            color=colour,
            alpha=alpha,
            width=0.6,
            edgecolor="white",
            linewidth=0.6,
            label=t(
                "paper.label.bucket_kind",
                bucket=t(f"bucket.{bucket}"),
                kind=t(f"paper.kind.{kind}"),
            ),
        )
        bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
    for i, total in enumerate(bottoms):
        ax.text(
            i,
            total,
            t("paper.label.million", value=format_number(total, 1)),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK_MUTED,
        )
    ax.set_ylim(0, max(bottoms) * 1.12)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([t(f"condition.{c}") for c in CONDITIONS], fontsize=8.5)
    _title(ax, "paper.figure.tokens")
    ax.set_ylabel(t("paper.axis.million_tokens"))
    _legend(ax, below=False)
    return _save(fig, out, "14_tokens_by_bucket")


def fig_calls(runs: list[RunSeries], out: Path) -> list[Path]:
    """LLM calls per run, agents and Director."""
    plt = _figure()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _style(ax)
    grouped = by_condition(runs)
    bottoms = [0.0] * len(CONDITIONS)
    for bucket, colour in (("agent", "#2a78d6"), ("director", "#7a4fb5")):
        heights = [_median([r.calls.get(bucket, 0) for r in grouped[c]]) for c in CONDITIONS]
        ax.bar(
            range(len(CONDITIONS)),
            heights,
            bottom=bottoms,
            color=colour,
            width=0.6,
            edgecolor="white",
            linewidth=0.6,
            label=t(f"bucket.{bucket}"),
        )
        bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
    for i, total in enumerate(bottoms):
        ax.text(
            i, total, format_number(total), ha="center", va="bottom", fontsize=8.5, color=INK_MUTED
        )
    ax.set_ylim(0, max(bottoms) * 1.12)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([t(f"condition.{c}") for c in CONDITIONS], fontsize=8.5)
    _title(ax, "paper.figure.calls")
    ax.set_ylabel(t("paper.axis.calls_per_run"))
    _legend(ax, below=False)
    return _save(fig, out, "15_llm_calls")


def fig_activations(runs: list[RunSeries], out: Path) -> list[Path]:
    """Activations per run: effective, wasted, plus invalid outputs."""
    plt = _figure()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _style(ax)
    grouped = by_condition(runs)
    effective = [_median([r.activations - r.wasted for r in grouped[c]]) for c in CONDITIONS]
    wasted = [_median([r.wasted for r in grouped[c]]) for c in CONDITIONS]
    invalid = [_median([r.invalid for r in grouped[c]]) for c in CONDITIONS]
    xs = range(len(CONDITIONS))
    ax.bar(xs, effective, width=0.6, color="#2a78d6", label=t("paper.label.effective"))
    ax.bar(xs, wasted, bottom=effective, width=0.6, color="#c9c7c1", label=t("paper.label.wasted"))
    ax.scatter(xs, invalid, marker="D", s=30, color=INK, zorder=3, label=t("paper.label.invalid"))
    ax.set_xticks(list(xs))
    ax.set_xticklabels([t(f"condition.{c}") for c in CONDITIONS], fontsize=8.5)
    _title(ax, "paper.figure.activations")
    ax.set_ylabel(t("paper.axis.per_run"))
    _legend(ax, below=False)
    return _save(fig, out, "16_activations")


def fig_actions(runs: list[RunSeries], out: Path) -> list[Path]:
    """Share of each executed action type, per condition (pooled over runs)."""
    plt = _figure()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    _style(ax)
    grouped = by_condition(runs)
    palette = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#7a4fb5",
        "#d4a020",
        "#c2416b",
        "#4aa3c9",
        "#8a8a3c",
        "#c9c7c1",
    )
    present = [a for a in ACTIONS if any(r.actions.get(a) for r in runs)]
    for i, condition in enumerate(CONDITIONS):
        counts = Counter()
        for r in grouped[condition]:
            counts.update(r.actions)
        total = sum(counts[a] for a in present) or 1
        left = 0.0
        for k, action in enumerate(present):
            share = counts[action] / total
            ax.barh(
                i,
                share,
                left=left,
                color=palette[k % len(palette)],
                edgecolor="white",
                linewidth=0.6,
                label=t(f"action.{action}") if i == 0 else None,
            )
            if share >= 0.06:
                ax.text(
                    left + share / 2,
                    i,
                    t("paper.label.percent", value=format_number(share * 100)),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                )
            left += share
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([t(f"condition.{c}") for c in CONDITIONS], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    _title(ax, "paper.figure.actions")
    ax.set_xlabel(t("paper.axis.share"))
    _legend(ax, ncols=5)
    return _save(fig, out, "17_action_mix")


def fig_orchestration(runs: list[RunSeries], out: Path) -> list[Path]:
    """What drives activation: the Director's TASK types in A, the triggers in B and C."""
    plt = _figure()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
    grouped = by_condition(runs)
    ax = axes[0]
    _style(ax)
    tasks = [_median([r.tasks.get(k, 0) for r in grouped["A"]]) for k in TASKS]
    ax.barh(range(len(TASKS)), tasks, color=CONDITION_COLOURS["A"], height=0.6)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([t(f"task.{k}") for k in TASKS], fontsize=8)
    ax.invert_yaxis()
    _title(ax, "paper.figure.tasks")
    ax.set_xlabel(t("paper.axis.per_run"))
    ax = axes[1]
    _style(ax)
    height = 0.38
    for j, condition in enumerate(("B", "C")):
        values = [_median([r.triggers.get(k, 0) for r in grouped[condition]]) for k in TRIGGERS]
        ax.barh(
            [i + (j - 0.5) * height for i in range(len(TRIGGERS))],
            values,
            height=height * 0.95,
            color=CONDITION_COLOURS[condition],
            label=t(f"condition.{condition}"),
        )
    ax.set_yticks(range(len(TRIGGERS)))
    ax.set_yticklabels([t(f"trigger.{k}") for k in TRIGGERS], fontsize=8)
    ax.invert_yaxis()
    _title(ax, "paper.figure.triggers")
    ax.set_xlabel(t("paper.axis.per_run"))
    _legend(ax, below=False)
    return _save(fig, out, "18_tasks_and_triggers")


def _km(ax, curves: dict[str, Any]) -> None:
    for condition in CONDITIONS:
        curve = curves.get(condition)
        if not curve or not curve.get("times"):
            continue
        ax.step(
            [0, *curve["times"]],
            [1.0, *curve["survival"]],
            where="post",
            linewidth=2.0,
            color=CONDITION_COLOURS[condition],
            linestyle=CONDITION_LINESTYLES[condition],
            label=t(
                "paper.label.km",
                condition=t(f"condition.{condition}"),
                events=curve["events"],
                n=curve["n"],
            ),
        )
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(t("report.axis.recovery_ticks"))
    ax.set_ylabel(t("report.axis.survival"))


def fig_survival(results: dict[str, Any], out: Path, good: str, product: str) -> list[Path]:
    """Kaplan-Meier: share of runs not yet recovered, the target and its product."""
    plt = _figure()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key, item in (
        (axes[0], "survival", good),
        (axes[1], "survival_propagation", product),
    ):
        _style(ax)
        _km(ax, (results.get(key) or {}).get("kaplan_meier", {}))
        ax.set_title(good_label(item), fontsize=10, loc="left", color=INK)
        _legend(ax, below=False)
    fig.suptitle(t("paper.figure.survival"), x=0.02, ha="left", fontsize=11, color=INK)
    return _save(fig, out, "19_recovery_kaplan_meier")


def fig_recovery_grid(results: dict[str, Any], out: Path, good: str, product: str) -> list[Path]:
    """Every run's recovery: the time when it recovered, a hatch when censored."""
    plt = _figure()
    runs = results.get("runs", [])
    seeds = sorted({r["seed"] for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("rec", ["#1baf7a", "#f2e6b3", "#eb6834"])
    horizon = max(((r.get("recovery") or {}).get("value") or 0 for r in runs), default=150) or 150
    for ax, key, item in ((axes[0], "recovery", good), (axes[1], "propagation_recovery", product)):
        ax.set_xlim(-0.5, max(1, len(seeds)) - 0.5)
        ax.set_ylim(-0.5, len(CONDITIONS) - 0.5)
        for i, condition in enumerate(CONDITIONS):
            for j, seed in enumerate(seeds):
                run = next(
                    (r for r in runs if r["seed"] == seed and r["condition"] == condition), None
                )
                rec = (run or {}).get(key) or {}
                value = rec.get("value")
                if value is None:
                    ax.add_patch(_rect(j, i, "#ffffff", hatch="///"))
                    ax.text(
                        j,
                        i,
                        t("report.value.missing"),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=INK_MUTED,
                    )
                elif rec.get("censored"):
                    ax.add_patch(_rect(j, i, "#e6e5e1", hatch="xx"))
                    ax.text(
                        j,
                        i,
                        t("paper.label.censored_at", value=format_number(value)),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=INK,
                    )
                else:
                    ax.add_patch(_rect(j, i, cmap(min(1.0, value / horizon))))
                    ax.text(
                        j,
                        i,
                        format_number(value),
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=INK,
                    )
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(s) for s in seeds], rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(CONDITIONS)))
        ax.set_yticklabels([t(f"condition.{c}") for c in CONDITIONS], fontsize=8)
        ax.invert_yaxis()
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0, colors=INK_MUTED)
        ax.set_title(good_label(item), fontsize=10, loc="left", color=INK)
    fig.suptitle(t("paper.figure.recovery_grid"), x=0.02, ha="left", fontsize=11, color=INK)
    fig.text(0.02, 0.01, t("paper.note.recovery_grid"), fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return _save(fig, out, "20_recovery_per_seed", layout=False)


def _rect(x: float, y: float, colour: Any, *, hatch: str | None = None):
    from matplotlib.patches import Rectangle

    return Rectangle(
        (x - 0.46, y - 0.42),
        0.92,
        0.84,
        facecolor=colour,
        edgecolor="#c9c7c1",
        linewidth=0.6,
        hatch=hatch,
    )


#: Per-run metrics from results.json drawn as paired per-seed plots: key path,
#: label key, and whether the value is a phase dict to read at "shock".
PAIRED_METRICS = (
    (("volatility_phase", "shock"), "paper.metric.volatility_shock"),
    (("useful_production", "useful_fraction"), "paper.metric.useful_production"),
    (("coordination", "coordination_efficiency"), "paper.metric.coordination_efficiency"),
    (("gini",), "paper.metric.gini"),
    (("tokens", "authoritative_total"), "paper.metric.tokens_millions"),
    (("volume_value", "shock"), "paper.metric.traded_value_shock"),
    (("propagation", "mean_price", "shock"), "paper.metric.propagation_price"),
    (("stockouts", "by_phase", "shock"), "paper.metric.stockout_ticks"),
)


def _dig(run: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = run
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, dict):  # a phase-keyed value collapsed to its total
        value = value.get("value")
    try:
        return float(value) if value is not None else None
    except TypeError, ValueError:
        return None
