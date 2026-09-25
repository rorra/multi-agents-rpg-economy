"""Figures for the report (design v1.0 s11; IMPLEMENTATION.md section 2).

Every label resolves through ``i18n.t``, so the English and Spanish figures are
rendered from one results file rather than drawn twice.

Conditions A, B and C are a categorical encoding -- identity, not magnitude --
so hues are assigned in a fixed order and never cycled or reused for anything
else. The palette is checked for colour-vision deficiency (worst adjacent pair
dE 9.2 under deuteranopia). The aqua sits slightly under the 3:1 contrast
guideline against white, so the report always emits a table alongside and
identity is never carried by colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpg_economy.i18n import t

#: Fixed categorical assignment, validated for CVD. Never cycled, never reused.
CONDITION_COLOURS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}

#: A second encoding alongside hue. Two conditions can trace the same path --
#: B and C coincide exactly whenever the bulletin changes no decision -- and one
#: line then hides the other. Distinct dash patterns keep an overplotted series
#: visible, and carry identity for readers who cannot separate the hues.
CONDITION_LINESTYLES = {"A": "-", "B": "--", "C": (0, (1, 1.6))}
#: Ink, not series colour: text never wears the hue of a mark.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def _style(ax) -> None:
    """Recessive axes and grid: the data is the figure, the frame is not."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def price_trajectory(
    series_by_condition: dict[str, list[float | None]],
    *,
    good_label: str,
    shock_tick: int,
    reversion_tick: int,
    path: Path,
    traded_value: bool = False,
) -> Path:
    """The target good's price over time, one line per condition.

    Change over time with a small number of identities: a line chart, with the
    shock interval shaded so the reader sees when the perturbation was in force
    without a second axis. With ``traded_value`` the series is the traded value
    per window (the liquidity_loss target, design s11.3), and the title and
    axis say so.
    """
    plt = _figure()
    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=160)
    _style(ax)

    ax.axvspan(shock_tick, reversion_tick, color=GRID, alpha=0.7, linewidth=0)

    for condition in sorted(series_by_condition):
        series = series_by_condition[condition]
        xs = [i for i, v in enumerate(series) if v is not None]
        ys = [float(series[i]) for i in xs]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            linewidth=2.0,
            color=CONDITION_COLOURS.get(condition, INK_MUTED),
            linestyle=CONDITION_LINESTYLES.get(condition, "-"),
            label=t(f"condition.{condition}"),
            solid_capstyle="round",
        )

    if traded_value:
        ax.set_title(t("report.figure.traded_value"), fontsize=11, loc="left")
        ax.set_ylabel(t("axis.traded_value"))
    else:
        ax.set_title(t("report.figure.price", good=good_label), fontsize=11, loc="left")
        ax.set_ylabel(t("axis.price"))
    ax.set_xlabel(t("axis.tick"))
    # Below the axes, so the legend never covers the series it labels. A run
    # with no trades at all has no series to label.
    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            frameon=False,
            fontsize=9,
            labelcolor=INK_MUTED,
            ncols=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def paired_differences(differences: dict[int, float], *, metric_label: str, path: Path) -> Path:
    """Per-seed paired differences against a zero line.

    The design pairs by seed, so the faithful picture of a paired contrast is
    every pair's own difference -- not two summary bars, which would hide whether
    a median difference comes from a consistent shift or from a few outliers.
    """
    plt = _figure()
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=160)
    _style(ax)

    seeds = sorted(differences)
    values = [differences[s] for s in seeds]
    ax.axhline(0, color=INK_MUTED, linewidth=1.0)
    ax.vlines(
        range(len(seeds)),
        [min(0, v) for v in values],
        [max(0, v) for v in values],
        color=GRID,
        linewidth=2,
    )
    ax.scatter(
        range(len(seeds)),
        values,
        s=44,
        color=[CONDITION_COLOURS["A"] if v <= 0 else CONDITION_COLOURS["B"] for v in values],
        zorder=3,
        edgecolor="white",
        linewidth=1.2,
    )

    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], rotation=45, ha="right")
    ax.set_title(t("report.figure.paired", metric=metric_label), fontsize=11, loc="left")
    ax.set_xlabel(t("axis.seed"))
    ax.set_ylabel(t("report.axis.difference"))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def kaplan_meier_curves(curves: dict[str, dict[str, Any]], *, path: Path) -> Path:
    """Step curves of recovery, censored observations respected.

    A survival curve is the right summary when some runs never recovered: the
    line simply does not reach zero, which is exactly what a censored
    observation means and what a bar of "mean recovery time" would conceal.
    """
    plt = _figure()
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=160)
    _style(ax)

    for condition in sorted(curves):
        curve = curves[condition]
        times, survival = curve["times"], curve["survival"]
        if not times:
            continue
        ax.step(
            [0, *times],
            [1.0, *survival],
            where="post",
            linewidth=2.0,
            color=CONDITION_COLOURS.get(condition, INK_MUTED),
            linestyle=CONDITION_LINESTYLES.get(condition, "-"),
            label=f"{t(f'condition.{condition}')} ({curve['events']}/{curve['n']})",
        )

    ax.set_ylim(-0.02, 1.02)
    ax.set_title(t("report.figure.km"), fontsize=11, loc="left")
    ax.set_xlabel(t("report.axis.recovery_ticks"))
    ax.set_ylabel(t("report.axis.survival"))
    if ax.get_legend_handles_labels()[0]:  # empty when no run's recovery was measurable
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
