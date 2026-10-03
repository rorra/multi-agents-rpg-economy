"""Render a replay bundle as one self-contained HTML page.

Tech choice: a single HTML file with vanilla JS and a canvas chart. No server,
no framework, no CDN, no build step. It opens from disk, works with no network,
and does not depend on a pinned framework version. The design calls this an
observation tool, and a page that inlines its own data is the smallest thing
that meets the bar.

i18n: every visible string is resolved through the catalogs here, at render
time, and injected as a lookup table. The stored logs stay language-neutral; the
same bundle renders to Spanish and English. Templates with placeholders are
passed through unfilled and completed client-side.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from rpg_economy.dashboard.replay import ReplayBundle
from rpg_economy.domain import ActionType, MessageType, RejectionCode, Scenario, TaskType
from rpg_economy.i18n import get_catalog, set_locale, t

ASSETS = Path(__file__).parent / "assets"

#: Enum families the page needs labels for. Listed explicitly so a missing
#: catalog entry fails at render time, in the language being built, rather than
#: showing a raw enum.
GOOD_KEYS = (
    ("resource", ("iron", "wood", "leather", "herbs", "arcane_crystal")),
    (
        "product",
        ("melee_weapon", "armor", "bow_gear", "staff_focus", "health_potion", "mana_potion"),
    ),
)
# From the enums, so a new action or type cannot reach the page unlabelled.
ACTION_KEYS = tuple(a.value for a in ActionType)
MESSAGE_KEYS = tuple(m.value for m in MessageType)
TASK_KEYS = tuple(task.value for task in TaskType)
TRIGGER_KEYS = (
    "T1_message_received",
    "T2_order_event",
    "T3_price_move",
    "T4_own_state_threshold",
    "T5_inactivity",
)
#: Why an action was refused or an activation wasted: the engine's closed codes,
#: plus the ones the log states elsewhere (rest after an expedition, the K limit,
#: the budget, the fallback WAIT). Free text that matches none is shown as is.
REJECTION_KEYS = tuple(
    dict.fromkeys(
        [
            *(code.value for code in RejectionCode),
            "resting",
            "k_limit",
            "budget_exhausted",
            "invalid_output",
            "wait",
        ]
    )
)
BUCKET_KEYS = ("agent", "director", "p2p_content", "bulletin")
ROLE_KEYS = ("adventurer", "artisan")
SPEC_KEYS = (
    ("subclass", ("warrior", "mage", "ranger")),
    ("specialization", ("smithing", "woodworking", "alchemy")),
)
FEED_EVENT_KEYS = (
    "tax_paid",
    "shock_applied",
    "shock_reverted",
    "budget_exhausted",
    "trade_executed",
    "expedition_resolved",
    "production_completed",
    "equipment_broken",
    "director_error",
    "order_expired",
)


def _strings(locale: str) -> dict[str, Any]:
    """Every label the page uses, resolved once."""
    catalog = get_catalog(locale)
    goods = {name: t(f"{prefix}.{name}") for prefix, names in GOOD_KEYS for name in names}
    specs = {name: t(f"{prefix}.{name}") for prefix, names in SPEC_KEYS for name in names}
    return {
        "locale": locale,
        # Every replay.* message as an unfilled template; the page fills them.
        # Sorted: the key set is a frozenset, and the page must be byte-reproducible.
        "tpl": {
            key: catalog.raw(key)
            for key in sorted(catalog.message_keys())
            if key.startswith("replay.")
        },
        "phase": {name: t(f"phase.{name}") for name in ("base", "shock", "recovery")},
        "condition": {c: t(f"condition.{c}") for c in ("A", "B", "C")},
        "scenario": {s.value: t(f"scenario.{s.value}") for s in Scenario},
        "good": goods,
        "role": {name: t(f"role.{name}") for name in ROLE_KEYS},
        "spec": specs,
        "action": {name: t(f"action.{name}") for name in ACTION_KEYS},
        "message": {name: t(f"message.{name}") for name in MESSAGE_KEYS},
        "task": {name: t(f"task.{name}") for name in TASK_KEYS},
        "trigger": {name: t(f"trigger.{name}") for name in TRIGGER_KEYS},
        "priority": {name: t(f"priority.{name}") for name in ("high", "medium", "low")},
        "bucket": {name: t(f"bucket.{name}") for name in BUCKET_KEYS},
        "event": {name: t(f"event.{name}") for name in FEED_EVENT_KEYS},
        "rejection": {name: t(f"rejection.{name}") for name in REJECTION_KEYS},
    }


_SCRIPT_ESCAPES = str.maketrans({"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"})


def _script_json(text: str) -> str:
    """Make serialized JSON safe to embed in a <script> element."""
    return text.translate(_SCRIPT_ESCAPES)


def render_replay(bundle: ReplayBundle, *, locale: str, focus_good: str | None = None) -> str:
    """Render the whole replay as one HTML document."""
    set_locale(locale)
    strings = _strings(locale)

    payload = bundle.to_payload()
    payload["focusGood"] = focus_good
    # Two tracks of the same condition (a model comparison) would otherwise
    # carry identical column headings.
    conditions = [track.condition for track in bundle.tracks]
    payload["labelByModel"] = len(set(conditions)) < len(conditions)

    head = bundle.tracks[0]
    title = t("replay.title", scenario=t(f"scenario.{head.scenario}"), seed=head.seed)

    css = (ASSETS / "replay.css").read_text(encoding="utf-8")
    icons = (ASSETS / "goods.svg").read_text(encoding="utf-8")
    script = (ASSETS / "replay.js").read_text(encoding="utf-8")

    # The data carries model-written text (short_reason), so it must not be
    # able to open a comment or close the script tag: every <, > and & is
    # written as a JSON unicode escape, which leaves the value identical.
    data = _script_json(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    strings_json = _script_json(json.dumps(strings, ensure_ascii=False))

    # Naming one model in the header of a two-model comparison is a false label.
    models = t("replay.meta.versus").join(
        dict.fromkeys(track.model_observed for track in bundle.tracks)
    )

    # A mock-answered run looks exactly like a model-answered one, so the page
    # says so even when whoever built the bundle gave no caveat.
    caveats = [bundle.caveat] if bundle.caveat else []
    if bundle.mock:
        caveats.append(t("replay.banner.mock"))
    banner = (
        f'<div class="banner"><strong>{t("replay.banner.label")}</strong> '
        f"{escape(' '.join(caveats))}</div>"
        if caveats
        else ""
    )
    provenance = "".join(
        "<li>"
        + escape(
            t(
                "replay.provenance.track",
                condition=t(f"condition.{track.condition}"),
                model=track.model_observed,
                prompts=track.prompt_version,
                config=track.config_hash[:12],
                # track.horizon is the last tick; the header counts ticks.
                horizon=track.horizon + 1,
            )
        )
        + f' <span class="runid">{escape(track.run_id)}</span></li>'
        for track in bundle.tracks
    )

    speeds = "".join(
        f'<option value="{v}"{" selected" if v == 4 else ""}>{v}&times;</option>'
        for v in (1, 2, 4, 8, 16, 32)
    )

    return f"""<!doctype html>
<html lang="{locale}" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>{css}</style>
</head>
<body>
{icons}
<header class="masthead">
  <div class="masthead-main">
    <h1>{escape(title)}</h1>
    <dl class="facts">
      <div><dt>{t("replay.meta.experiment")}</dt><dd>{t(f"experiment.{head.experiment}")}</dd></div>
      <div><dt>{t("replay.meta.model")}</dt><dd>{escape(models)}</dd></div>
      <div><dt>{t("replay.meta.horizon")}</dt><dd>{t("replay.meta.horizon_value", ticks=bundle.horizon + 1)}</dd></div>
    </dl>
  </div>
  <div class="masthead-tools">
    <details class="provenance">
      <summary>{t("replay.provenance.toggle")}</summary>
      <ul>{provenance}</ul>
    </details>
    <button id="theme" class="icon" type="button" title="{t("replay.theme")}" aria-label="{t("replay.theme")}">◐</button>
  </div>
</header>
{banner}
<nav class="transport" aria-label="{t("replay.aria.transport")}">
  <div class="transport-row">
    <button id="play" class="play" type="button">{t("replay.play")}</button>
    <button id="back" class="icon" type="button" title="{t("replay.back")}" aria-label="{t("replay.back")}">&lsaquo;</button>
    <button id="fwd" class="icon" type="button" title="{t("replay.forward")}" aria-label="{t("replay.forward")}">&rsaquo;</button>
    <div class="clock"><span id="tickbox"></span><span id="cyclebox"></span></div>
    <span class="phase" id="phasetag"></span>
    <div class="scrubwrap"><input type="range" id="scrub" min="0" value="0" step="1" aria-label="{t("replay.aria.tick")}"><div class="scrub-marks" id="marks"></div></div>
    <label class="speed">{t("replay.speed")}<select id="speed">{speeds}</select></label>
  </div>
  <div class="transport-row secondary">
    <button id="shock" type="button">{t("replay.jump_shock")}</button>
    <button id="recovery" type="button">{t("replay.jump_recovery")}</button>
    <button id="taxjump" type="button" disabled>{t("replay.jump_tax")}</button>
    <button id="restart" type="button">{t("replay.restart")}</button>
  </div>
</nav>

<main>
  <section class="shockbox" id="shockbox" hidden></section>

  <section class="overview">
    <div class="chartcard">
      <div class="chart-head">
        <h2 id="charttitle"></h2>
        <label class="goodpick">{t("replay.good")}<select id="good"></select></label>
      </div>
      <canvas id="chart" role="img" aria-label="{t("replay.aria.price_chart")}"></canvas>
      <h3 class="stock-head" id="stocktitle"></h3>
      <canvas id="stockchart" role="img" aria-label="{t("replay.aria.stock_chart")}"></canvas>
      <p class="hint">{t("replay.chart.hint")} {t("replay.chart.marks")} {t("replay.stock.legend")}</p>
    </div>
    <div class="nowcard">
      <h2 id="nowtitle"></h2>
      <table class="now" id="now"></table>
    </div>
  </section>

  <section class="columns" id="grid"></section>

  <section class="guide" id="guide" hidden></section>
</main>

<footer>
  <p>{t("replay.footer.readonly")}</p>
  <p>{t("replay.footer.keys")}</p>
</footer>

<script>window.DATA={data};window.STR={strings_json};</script>
<script>{script}</script>
</body>
</html>
"""


def write_replay(
    bundle: ReplayBundle, destination: Path | str, *, locale: str, focus_good: str | None = None
) -> Path:
    """Write the replay page for one language."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_replay(bundle, locale=locale, focus_good=focus_good), encoding="utf-8")
    return path
